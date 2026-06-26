import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from reproducibility.realdata_prediction.prediction_comparison import clr_krr, clr_svm

MASTER_SEED = 20241225


def load_ileum():
    counts = pd.read_csv("./datasets/MLRepo/gevers/refseq/taxatable.txt", sep="\t")
    y_df = pd.read_csv("./datasets/MLRepo/gevers/task-ileum.txt", sep="\t")
    counts = counts[y_df["#SampleID"]]
    counts = counts[(counts > 0).sum(1) >= 5]

    x = counts.T.to_numpy().astype(np.float64)
    for i in range(x.shape[0]):
        x[i] = np.where(x[i] == 0, x[i][x[i] > 0].min() * 0.5, x[i])

    y = y_df["Var"].to_numpy().flatten()
    y_vals = sorted(set(y))
    y = np.array([-1 if val == y_vals[0] else 1 for val in y], dtype=int)
    return x, y


def load_vaginal():
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
    outdir = Path("results/realdata_prediction")
    outdir.mkdir(parents=True, exist_ok=True)

    x_ileum, y_ileum = load_ileum()
    ileum_result = {
        "dataset": "ileum",
        "method": "clr-Kernel (SVM)",
        "metric": "accuracy",
        "values": clr_svm(x_ileum, y_ileum, reps=100, seed=MASTER_SEED),
        "seed": MASTER_SEED,
    }
    with (outdir / "ileum_clr-Kernel.pkl").open("wb") as f:
        pickle.dump(ileum_result, f)

    x_vaginal, y_vaginal = load_vaginal()
    vaginal_result = {
        "dataset": "vaginal",
        "method": "clr-Kernel (KRR)",
        "metric": "mse",
        "values": clr_krr(x_vaginal, y_vaginal, reps=100, seed=MASTER_SEED),
        "seed": MASTER_SEED,
    }
    with (outdir / "vaginal_clr-Kernel.pkl").open("wb") as f:
        pickle.dump(vaginal_result, f)


if __name__ == "__main__":
    main()
