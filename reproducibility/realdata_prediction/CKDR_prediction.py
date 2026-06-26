import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from functions.cross_val import ckdr_cv_parallel

MASTER_SEED = 20241225


def load_data(dataset):
    y_names = {"gevers": "ileum", "ravel": "nugent-score"}
    counts = pd.read_csv(f"./datasets/MLRepo/{dataset}/refseq/taxatable.txt", sep="\t")
    y_df = pd.read_csv(f"./datasets/MLRepo/{dataset}/task-{y_names[dataset]}.txt", sep="\t")

    counts = counts[y_df["#SampleID"]]
    counts = counts[(counts > 0).sum(1) >= 5]

    x = counts.to_numpy().T
    x = x / x.sum(1)[:, None]
    y = y_df["Var"].to_numpy().flatten()

    if dataset == "gevers":
        y_vals = sorted(set(y))
        y = np.array([-1.0 if val == y_vals[0] else 1.0 for val in y])
        return x, y, "binary", "accuracy", "ileum"

    return x, y.astype(float), None, "mse", "vaginal"


def parse_dim_setting(dim_setting):
    if dim_setting == "3-7":
        return [3, 4, 5, 6, 7], "3-7"
    dim = int(dim_setting)
    return [dim], str(dim)


def prediction_measure(model, P, sigma, epsilon, x, y, metric):
    x_p, y_p = model.test_processing(x, y)
    pred = model.predict(P, sigma, epsilon, x_p)

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(y_p):
        y_p = y_p.detach().cpu().numpy()

    if metric == "accuracy":
        return float(np.mean(pred == y_p.ravel()))

    std = model.Y_std.detach().cpu().numpy() if torch.is_tensor(model.Y_std) else model.Y_std
    return float(np.mean((pred * std - y_p * std) ** 2))


def run_one_split(run_index, x, y, type_y, metric, dim_list, dim_label, dataset_label, args):
    out_file = Path(args.outdir) / f"{dataset_label}_dim{dim_label}_run{run_index:03d}.pkl"
    if out_file.exists() and not args.overwrite:
        print(f"{out_file}: skip existing", flush=True)
        return str(out_file)

    seed = MASTER_SEED + run_index
    split_rs = np.random.RandomState(seed)
    indices = np.arange(x.shape[0])
    stratify = y if type_y == "binary" else None
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, stratify=stratify, random_state=split_rs
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cv_result = ckdr_cv_parallel(
        x_train,
        y_train,
        type_Y=type_y,
        folds=args.folds,
        dim_list=dim_list,
        epsilon_list=np.geomspace(1e-4, 0.01, 10),
        sigma_list=[1.0],
        med=True,
        solver="sca",
        seed=split_rs,
        n_jobs=args.n_jobs_cv,
        verbose=args.verbose,
        max_iter=args.max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        inner_tol=args.inner_tol,
    )

    sigma = cv_result["parameters"]["sigma_Z"]
    epsilon = cv_result["parameters"]["epsilon"]
    model = cv_result["CV_ckdr_class"]
    P = cv_result["CV_fitted_P"]
    train_value = prediction_measure(model, P, sigma, epsilon, x_train, y_train, metric)
    test_value = prediction_measure(model, P, sigma, epsilon, x_test, y_test, metric)

    record = {
        "run": run_index,
        "seed": seed,
        "category": "ckdr",
        "metric": metric,
        "train_metric": train_value,
        "test_metric": test_value,
        "parameters": cv_result["parameters"],
        "best_index": cv_result["best_index"],
        "cv_result": cv_result,
        "config": {
            "dataset": dataset_label,
            "dim_label": dim_label,
            "dim_list": dim_list,
            "epsilon_grid": np.geomspace(1e-4, 0.01, 10),
            "sigma_list": [1.0],
            "med": True,
            "selection_rule": "argmin",
            "warm_start": False,
            "train_indices": train_idx,
            "test_indices": test_idx,
        },
    }
    if metric == "accuracy":
        record["train_acc"] = train_value
        record["test_acc"] = test_value
    else:
        record["train_mse"] = train_value
        record["test_mse"] = test_value

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        pickle.dump(record, f)

    print(f"{out_file}: m={record['parameters']['target_dim']}, eps={epsilon:.6g}, test={test_value:.4f}", flush=True)
    return str(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gevers", "ravel"], required=True)
    parser.add_argument("--dim-setting", choices=["3", "5", "3-7"], required=True)
    parser.add_argument("--start-run", type=int, default=0)
    parser.add_argument("--end-run", type=int, default=100)
    parser.add_argument("--outdir", default="results/realdata_prediction/ckdr")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_jobs_cv", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--inner_max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--inner_tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    x, y, type_y, metric, dataset_label = load_data(args.dataset)
    dim_list, dim_label = parse_dim_setting(args.dim_setting)

    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one_split)(run, x, y, type_y, metric, dim_list, dim_label, dataset_label, args)
        for run in range(args.start_run, args.end_run)
    )


if __name__ == "__main__":
    main()
