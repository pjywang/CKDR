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
    repeat_clr_kernel_krr,
    repeat_clr_kernel_svm,
    repeat_clr_rf_clf,
    repeat_clr_rf_reg,
    repeat_lc_lasso_clf,
    repeat_lc_lasso_reg,
)


MASTER_SEED = 20241225


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["lc_lasso", "clr_kernel", "clr_rf"])
    parser.add_argument("--n", type=int, required=True, choices=[200, 500, 1000])
    parser.add_argument("--y_func", required=True, choices=["Y1", "Y2", "Y3", "Y4"])
    parser.add_argument("--n_jobs", type=int, default=-2)
    args = parser.parse_args()

    y_func = {"Y1": Y1, "Y2": Y2, "Y3": Y3, "Y4": Y4}[args.y_func]
    is_binary = args.y_func in ["Y3", "Y4"]

    if args.method == "lc_lasso":
        runner = repeat_lc_lasso_clf if is_binary else repeat_lc_lasso_reg
        runner(args.n, 100, y_func, logistic_normal, njobs=args.n_jobs,
               reps=100, seed=MASTER_SEED)
    elif args.method == "clr_kernel":
        if is_binary:
            repeat_clr_kernel_svm(args.n, 100, y_func, logistic_normal,
                                  njobs=args.n_jobs, reps=100, seed=MASTER_SEED)
        else:
            repeat_clr_kernel_krr(args.n, 100, y_func, logistic_normal,
                                  reps=100, seed=MASTER_SEED)
    elif args.method == "clr_rf":
        runner = repeat_clr_rf_clf if is_binary else repeat_clr_rf_reg
        runner(args.n, 100, y_func, logistic_normal, njobs=args.n_jobs,
               reps=100, seed=MASTER_SEED)


if __name__ == "__main__":
    main()
