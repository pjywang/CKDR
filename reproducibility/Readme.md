# Reproducibility report

The jupyter notebook files in this folder reproduces the experiments in the paper. Specifically,
- `realdata-visualizations.ipynb`: Produces dual visualization from our CKDR method.
- `realdata-predictions.ipynb`: Prediction performance comparison with competing methods for compositional data (their codes are included in `./other_methods` folder)
- `simulation_results.ipynb`: Loads simulation result files, displays summary data frames, and exports paper tables.
- `simulation/`: Command-line scripts and helpers for regenerating simulation result files.
- `realdata_prediction/`: Command-line scripts and helpers for regenerating real-data prediction result files.

Other `.py` files are shared notebook/table helpers, which should not be removed.
