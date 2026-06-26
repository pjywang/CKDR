from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score

from functions import KQuantiles


RESULT_ROOT = Path("results/simulation")
TABLE_DIR = Path("results/tables")
N_LIST = [200, 500, 1000]
Y_LIST = ["Y1", "Y2", "Y3", "Y4"]
SETTING_LABEL = {"Y1": "(i)", "Y2": "(ii)", "Y3": "(iii)", "Y4": "(iv)"}
MASTER_SEED = 20241225


def _load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _mean_se(values):
    values = np.asarray(values, dtype=float).ravel()
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def _get_true(p):
    true = np.zeros(p)
    cut1 = p // 5
    cut2 = p // 2
    true[cut1:cut2] = 1
    true[cut2:] = 2
    return true


def _true_mat(p, y_name):
    target_dim = 2 if y_name in ["Y1", "Y3"] else 3
    mat = np.zeros((target_dim, p))
    cut1 = p // 5
    cut2 = p // 2

    if target_dim == 3:
        mat[0, :cut1] = 1
        mat[1, cut1:cut2] = 1
        mat[2, cut2:] = 1
    elif y_name == "Y1":
        mat[0, :cut1] = 1
        mat[1, cut2:] = 1
        mat[0, cut1:cut2] = 4 / 9
        mat[1, cut1:cut2] = 5 / 9
    elif y_name == "Y3":
        mat[0, cut2:] = 1
        mat[1, cut1:cut2] = 1
        mat[0, :cut1] = 5 / 8
        mat[1, :cut1] = 3 / 8
    return mat


def _subsp_proj(mat):
    return mat.T @ np.linalg.pinv(mat @ mat.T) @ mat


def _subsp_dist(mat_a, mat_b):
    rank_a = np.linalg.matrix_rank(mat_a)
    rank_b = np.linalg.matrix_rank(mat_b)
    sq_frob = np.sum((_subsp_proj(mat_a) - _subsp_proj(mat_b)) ** 2)
    sq_dist = (sq_frob - abs(rank_a - rank_b)) / (2 * min(rank_a, rank_b))
    return np.sqrt(max(sq_dist, 0.0))


def _beta_to_cdr(beta):
    beta = np.asarray(beta, dtype=float).ravel()
    gap = beta.max() - beta.min()
    mat = np.zeros((2, len(beta)))
    mat[0] = (beta.max() - beta) / gap
    mat[1] = (beta - beta.min()) / gap
    return mat


def load_subspace_results():
    rows = []
    for y_name in Y_LIST:
        for n in N_LIST:
            for m, method in [(None, r"CKDR-$m^\star$"), ("cv", r"CKDR$^*$")]:
                tag = "None" if m is None else "cv"
                path = RESULT_ROOT / "subsp_convergence" / f"op_{n}_{y_name}_logistic_normal{tag}.pickle"
                arr = np.asarray(_load_pickle(path), dtype=float)
                dist_mean, dist_se = _mean_se(arr[:, 0] * 100)
                ari_mean, ari_se = _mean_se(arr[:, 1] * 100)
                rows.append({
                    "setting": SETTING_LABEL[y_name], "Y": y_name, "n": n,
                    "method": method,
                    "rho_mean": dist_mean, "rho_se": dist_se,
                    "ari_mean": ari_mean, "ari_se": ari_se,
                    "rank_deficient": int(np.sum(arr[:, 2] < (2 if y_name in ["Y1", "Y3"] else 3))),
                })

            if y_name in ["Y1", "Y2"]:
                mat = loadmat(RESULT_ROOT / "rs_es_results" / f"beta_rs_n{n}_{y_name}.mat")
                beta_all = mat["beta_all"]
                true_cluster = _get_true(100)
                true_matrix = _true_mat(100, y_name)
                distances, aris = [], []
                for i in range(beta_all.shape[0]):
                    p_hat = _beta_to_cdr(beta_all[i, 0])
                    if y_name == "Y1":
                        distances.append(_subsp_dist(p_hat, true_matrix))
                    rs = np.random.RandomState(MASTER_SEED + i)
                    clus = KQuantiles(n_clusters=3, random_state=rs, verbose=False)
                    clus.fit(p_hat.T)
                    aris.append(adjusted_rand_score(true_cluster, clus.clusters))
                dist_mean, dist_se = (np.nan, np.nan)
                if y_name == "Y1":
                    dist_mean, dist_se = _mean_se(np.asarray(distances) * 100)
                ari_mean, ari_se = _mean_se(np.asarray(aris) * 100)
                rows.append({
                    "setting": SETTING_LABEL[y_name], "Y": y_name, "n": n,
                    "method": "RS-ES",
                    "rho_mean": dist_mean, "rho_se": dist_se,
                    "ari_mean": ari_mean, "ari_se": ari_se,
                    "rank_deficient": np.nan,
                })

            if y_name in ["Y3", "Y4"]:
                arr = np.asarray(_load_pickle(
                    RESULT_ROOT / "amalgam_results" / f"op_{n}_{y_name}.pickle"
                ), dtype=float)
                true_cluster = _get_true(100)
                true_matrix = _true_mat(100, y_name)
                distances, aris = [], []
                for i, p_hat in enumerate(arr):
                    distances.append(_subsp_dist(p_hat, true_matrix))
                    rs = np.random.RandomState(MASTER_SEED + i)
                    clus = KQuantiles(n_clusters=3, random_state=rs, verbose=False)
                    clus.fit(p_hat.T)
                    aris.append(adjusted_rand_score(true_cluster, clus.clusters))
                dist_mean, dist_se = _mean_se(np.asarray(distances) * 100)
                ari_mean, ari_se = _mean_se(np.asarray(aris) * 100)
                rows.append({
                    "setting": SETTING_LABEL[y_name], "Y": y_name, "n": n,
                    "method": "Amalgam",
                    "rho_mean": dist_mean, "rho_se": dist_se,
                    "ari_mean": ari_mean, "ari_se": ari_se,
                    "rank_deficient": np.nan,
                })
    return pd.DataFrame(rows)


def load_prediction_results():
    rows = []
    methods = [
        (r"CKDR-$m^\star$", "predictions", "{n}_{Y}_None.pickle"),
        (r"CKDR$^*$", "predictions", "{n}_{Y}_cv.pickle"),
        ("LC-Lasso", "lc_lasso_results", "{n}_{Y}.pickle"),
        ("clr-Kernel", "clr_kernel_results", "{n}_{Y}.pickle"),
        ("clr-RF", "clr_rf_results", "{n}_{Y}.pickle"),
    ]

    for y_name in Y_LIST:
        metric = "MSE" if y_name in ["Y1", "Y2"] else "MCR"
        for n in N_LIST:
            for method, folder, template in methods:
                arr = np.asarray(_load_pickle(
                    RESULT_ROOT / folder / template.format(n=n, Y=y_name)
                ), dtype=float)
                if metric == "MCR":
                    arr = 1 - arr
                mean, se = _mean_se(arr)
                rows.append({
                    "metric": metric, "setting": SETTING_LABEL[y_name],
                    "Y": y_name, "n": n, "method": method,
                    "mean": mean, "se": se,
                })

            if y_name in ["Y1", "Y2"]:
                mat = loadmat(RESULT_ROOT / "rs_es_results" / f"mse_rs_n{n}_{y_name}.mat")
                mean, se = _mean_se(mat["mse_all"])
                rows.append({
                    "metric": metric, "setting": SETTING_LABEL[y_name],
                    "Y": y_name, "n": n, "method": "RS-ES",
                    "mean": mean, "se": se,
                })
    return pd.DataFrame(rows)


def _fmt(mean, se, digits=1, best=False, omit_leading_zero=False):
    if pd.isna(mean):
        return "--"
    text = f"{mean:.{digits}f} ({se:.{digits}f})"
    if omit_leading_zero:
        text = text.replace("0.", ".")
    return f"\\textbf{{{text}}}" if best else text


def export_subspace_table(path=TABLE_DIR / "simulation_subspace.tex"):
    df = load_subspace_results()
    methods_by_y = {
        "Y1": [r"CKDR-$m^\star$", r"CKDR$^*$", "RS-ES"],
        "Y2": [r"CKDR-$m^\star$", r"CKDR$^*$", "RS-ES"],
        "Y3": [r"CKDR-$m^\star$", r"CKDR$^*$", "Amalgam"],
        "Y4": [r"CKDR-$m^\star$", r"CKDR$^*$", "Amalgam"],
    }

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Simulation results on estimation accuracy for SDR and true amalgamation, with standard errors in parentheses. Bold-faced numbers indicate the best result for each setting.}",
        r"    \label{tab:simulation-subspace}",
        r"    \begin{scriptsize}",
        r"    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccccc}",
        r"    \toprule",
        r"     &  & \multicolumn{3}{c}{$\rho(\row(\wh P_n), \mathcal{C}_{Y|X}) \times 100$} & \multicolumn{3}{c}{ARI $\times 100$} \\",
        r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"    Setting& Method & $n=200$ & $n=500$ & $n=1000$ & $n=200$ & $n=500$ & $n=1000$ \\",
        r"    \midrule",
    ]

    for y_idx, y_name in enumerate(Y_LIST):
        setting = SETTING_LABEL[y_name]
        subset = df[df["Y"] == y_name]
        best_rho = {
            n: subset[(subset["n"] == n) & subset["rho_mean"].notna()]
            .sort_values("rho_mean").iloc[0]["method"]
            for n in N_LIST
        }
        best_ari = {
            n: subset[subset["n"] == n].sort_values("ari_mean", ascending=False).iloc[0]["method"]
            for n in N_LIST
        }
        methods = methods_by_y[y_name]
        for j, method in enumerate(methods):
            label = rf"\multirow{{3}}{{*}}{{{setting}}}" if j == 0 else ""
            row = subset[subset["method"] == method].set_index("n")
            rho = [
                _fmt(row.loc[n, "rho_mean"], row.loc[n, "rho_se"],
                     best=(best_rho[n] == method))
                if n in row.index else "--"
                for n in N_LIST
            ]
            ari = [
                _fmt(row.loc[n, "ari_mean"], row.loc[n, "ari_se"],
                     best=(best_ari[n] == method))
                if n in row.index else "--"
                for n in N_LIST
            ]
            lines.append("     " + " & ".join([label, method] + rho + ari) + r" \\")
        if y_idx != len(Y_LIST) - 1:
            lines.append(r"    \addlinespace")

    lines += [
        r"    \bottomrule",
        r"    \end{tabular*}",
        r"    \end{scriptsize}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return df, path


def export_prediction_table(path=TABLE_DIR / "simulation_predictions.tex"):
    df = load_prediction_results()
    methods = [r"CKDR-$m^\star$", r"CKDR$^*$", "LC-Lasso", "clr-Kernel", "clr-RF", "RS-ES"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Simulation results on prediction performance, measured by MSE for settings (i) and (ii), and MCR for (iii) and (iv). Standard errors are given in parentheses.}",
        r"\label{tab:simulation_predictions}",
        r"\scriptsize",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lllcccccc}",
        r"\toprule",
        r"Metric & Setting & $n$ & CKDR-$m^\star$ & CKDR$^*$ & LC-Lasso & clr-Kernel & clr-RF & RS-ES \\",
        r"\midrule",
    ]

    for metric_idx, metric in enumerate(["MSE", "MCR"]):
        y_group = ["Y1", "Y2"] if metric == "MSE" else ["Y3", "Y4"]
        for y_idx, y_name in enumerate(y_group):
            for n_idx, n in enumerate(N_LIST):
                subset = df[(df["Y"] == y_name) & (df["n"] == n)]
                best_method = subset.sort_values("mean").iloc[0]["method"]
                metric_label = rf"\multirow{{7}}{{*}}{{{metric}}}" if y_idx == 0 and n_idx == 0 else ""
                setting_label = rf"\multirow{{3}}{{*}}{{{SETTING_LABEL[y_name]}}}" if n_idx == 0 else ""
                entries = []
                row = subset.set_index("method")
                for method in methods:
                    if method in row.index:
                        entries.append(_fmt(row.loc[method, "mean"], row.loc[method, "se"],
                                            digits=3, best=(best_method == method),
                                            omit_leading_zero=True))
                    else:
                        entries.append("--")
                lines.append(" " + " & ".join([metric_label, setting_label, str(n)] + entries) + r" \\")
            if y_idx == 0:
                lines.append(r" \addlinespace")
        if metric_idx == 0:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return df, path
