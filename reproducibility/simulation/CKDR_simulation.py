import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from reproducibility.simulation.simulations import (
    Y1, Y2, Y3, Y4,
    logistic_normal,
    repeat_cv,
    repeat_prediction,
)


MASTER_SEED = 20241225


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=[200, 500, 1000])
    parser.add_argument("--y_func", required=True, choices=["Y1", "Y2", "Y3", "Y4"])
    parser.add_argument("--m", required=True, choices=["oracle", "cv"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    y_func = {"Y1": Y1, "Y2": Y2, "Y3": Y3, "Y4": Y4}[args.y_func]
    m = None if args.m == "oracle" else "cv"

    os.makedirs("results/simulation/subsp_convergence", exist_ok=True)
    os.makedirs("results/simulation/predictions", exist_ok=True)

    repeat_cv(
        args.n, 100, y_func, logistic_normal, m=m, Y_noise=0.1,
        njobs=args.n_jobs, reps=100, foldername="subsp_convergence",
        load=False, seed=MASTER_SEED, solver="sca",
    )
    repeat_prediction(
        args.n, 100, y_func, logistic_normal, m=m, Y_noise=0.1,
        njobs=args.n_jobs, reps=100, foldername="predictions",
        load=True, load_folder="subsp_convergence",
        seed=MASTER_SEED, solver="sca",
    )


if __name__ == "__main__":
    main()
