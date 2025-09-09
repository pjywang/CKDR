# CKDR: a nonparametric method for interpretable dimension reduction of compositional data

A Python code for interpretable dimension reduction of compositional data using column-stochastic matrices. The proposed CKDR method finds an optimal reduction matrix in terms of sufficient dimension reduction (SDR), a supervised dimension reduction technique. Detailed interpretability guide with the proposed method is provided in the paper:

Junyoung Park, Cheolwoo Park, and Jeongyoun Ahn. "Interpretable dimension reduction for compositional data" [arXiv 2509.05563](https://arxiv.org/abs/2509.05563).

If you have any questions, please feel free to reach out to: junyoup@umich.edu


## Dependencies
`numpy`, `pandas`, `scipy`, `sklearn`, `Pytorch >= 2.7.0`, `joblib` (for parallel cross-validation), `python-ternary` (for ternary visualizations; see [instructions](https://github.com/marcharper/python-ternary))


## Reproducibility
The files in the `reproducibility` folder reports how to reproduce the experiments provided in the paper.