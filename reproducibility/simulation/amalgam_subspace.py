import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from reproducibility.simulation.simulations import Y3, Y4, repeat_amalgam


MASTER_SEED = 20241225


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=[200, 500, 1000])
    parser.add_argument("--y_func", required=True, choices=["Y3", "Y4"])
    parser.add_argument("--n_jobs", type=int, default=-2)
    args = parser.parse_args()

    os.makedirs("results/simulation/amalgam_results", exist_ok=True)
    y_func = {"Y3": Y3, "Y4": Y4}[args.y_func]
    repeat_amalgam(args.n, 100, y_func, njobs=args.n_jobs,
                   reps=100, seed=MASTER_SEED,
                   foldername="amalgam_results")


if __name__ == "__main__":
    main()
