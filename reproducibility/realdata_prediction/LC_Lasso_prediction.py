import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from reproducibility.realdata_prediction.prediction_comparison import codalasso_clf, codalasso_reg

MASTER_SEED = 20241225


def load_ileum_counts():
    counts = pd.read_csv("./datasets/MLRepo/gevers/refseq/taxatable.txt", sep="\t")
    y_df = pd.read_csv("./datasets/MLRepo/gevers/task-ileum.txt", sep="\t")
    counts = counts[y_df["#SampleID"]]
    counts = counts[(counts > 0).sum(1) >= 5]

    for col_name in counts.columns:
        col = counts[col_name]
        counts[col_name] = col.where(col > 0, col[col > 0].min() * 0.5)

    y = y_df["Var"].to_numpy().flatten()
    y_vals = sorted(set(y))
    y = pd.Series(np.array([-1.0 if val == y_vals[0] else 1.0 for val in y]))
    return counts.T, y


def load_vaginal_counts():
    counts = pd.read_csv("./datasets/MLRepo/ravel/refseq/taxatable.txt", sep="\t")
    y_df = pd.read_csv("./datasets/MLRepo/ravel/task-nugent-score.txt", sep="\t")
    counts = counts[y_df["#SampleID"]]
    counts = counts[(counts > 0).sum(1) >= 5]

    x = counts.T.to_numpy().astype(np.float64)
    for i in range(x.shape[0]):
        x[i] = np.where(x[i] == 0, x[i][x[i] > 0].min() * 0.5, x[i])

    y = y_df["Var"].to_numpy().astype(float)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-3)
    args = parser.parse_args()

    outdir = Path("results/realdata_prediction")
    outdir.mkdir(parents=True, exist_ok=True)

    x_ileum, y_ileum = load_ileum_counts()
    ileum_values = codalasso_clf(
        x_ileum,
        y_ileum,
        cvfolds=5,
        reps=100,
        lamseq=np.geomspace(0.001, 1, 30),
        njobs=args.n_jobs,
        seed=MASTER_SEED,
    )
    with (outdir / "ileum_LC-Lasso.pkl").open("wb") as f:
        pickle.dump(
            {"dataset": "ileum", "method": "LC-Lasso", "metric": "accuracy", "values": ileum_values, "seed": MASTER_SEED},
            f,
        )

    x_vaginal, y_vaginal = load_vaginal_counts()
    vaginal_values, vaginal_lambda = codalasso_reg(
        x_vaginal,
        y_vaginal,
        cvfolds=5,
        reps=100,
        lamseq=np.geomspace(0.001, 1, 30),
        njobs=args.n_jobs,
        seed=MASTER_SEED,
    )
    with (outdir / "vaginal_LC-Lasso.pkl").open("wb") as f:
        pickle.dump(
            {
                "dataset": "vaginal",
                "method": "LC-Lasso",
                "metric": "mse",
                "values": vaginal_values,
                "selected_lambda": vaginal_lambda,
                "seed": MASTER_SEED,
            },
            f,
        )


if __name__ == "__main__":
    main()
