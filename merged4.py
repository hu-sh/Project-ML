from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PRED_CSV = Path("plots/errors/predictions_cv.csv")
FIT_PARAMS = Path("plots/errors/fit_params_by_fold.csv")
Z_MIN = -100.0
Z_MAX = 100.0
Z_GRID_SIZE = 100001


def _y_of_z(z):
    z = np.asarray(z)
    k1 = 0.5463
    k2 = 1.1395
    y1 = k1 * z * np.cos(k2 * z)
    y2 = k1 * z * np.sin(k2 * z)
    sum_y34 = -z * np.cos(2 * z)
    y3 = (sum_y34 + z) / 2.0
    y4 = (sum_y34 - z) / 2.0
    return np.column_stack([y1, y2, y3, y4])


def _mee(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def _load_fit_params_by_fold(path):
    df = pd.read_csv(path)
    params = {}
    for _, row in df.iterrows():
        params[int(row["fold"])] = {
            "gamma_k": row["gamma_k"],
            "gamma_loc": row["gamma_loc"],
            "gamma_scale": row["gamma_scale"],
            "t_est_raw_df": row["t_est_raw_df"],
            "t_est_raw_loc": row["t_est_raw_loc"],
            "t_est_raw_scale": row["t_est_raw_scale"],
            "norm_mu": row["norm_mu"],
            "norm_sigma": row["norm_sigma"],
            "t_pred_df": row["t_pred_df"],
            "t_pred_loc": row["t_pred_loc"],
            "t_pred_scale": row["t_pred_scale"],
        }
    return params


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {PRED_CSV}")
    if not FIT_PARAMS.exists():
        raise FileNotFoundError(f"Missing fit params: {FIT_PARAMS}")

    df = pd.read_csv(PRED_CSV)
    y_val = df[["y_val_1", "y_val_2", "y_val_3", "y_val_4"]].values
    y_final_pred = df[["y_pred_1", "y_pred_2", "y_pred_3", "y_pred_4"]].values
    z_val = df["z_val"].values
    z_pred = df["z_pred"].values
    z_pred_pca = df["z_pred_pca"].values
    z_est_raw = df["z_est_raw"].values

    params_by_fold = _load_fit_params_by_fold(FIT_PARAMS)

    folds = df["fold"].values.astype(int)
    z_grid = np.linspace(Z_MIN, Z_MAX, Z_GRID_SIZE)
    z_star = np.empty_like(z_val)
    loglik_star = np.empty_like(z_val)
    loglik_true = np.empty_like(z_val)
    for fold in np.unique(folds):
        mask = folds == fold
        params = params_by_fold[fold]
        idx = np.where(mask)[0]
        for i in idx:
            err_pred = z_pred[i] - z_grid
            err_pred_pca = z_pred_pca[i] - z_grid
            err_est_raw = z_est_raw[i] - z_grid
            y_c = _y_of_z(z_grid)
            err_y = np.linalg.norm(y_final_pred[i] - y_c, axis=1)

            loglik = (
                stats.t.logpdf(
                    err_pred,
                    df=params["t_pred_df"],
                    loc=params["t_pred_loc"],
                    scale=params["t_pred_scale"],
                )
                + stats.norm.logpdf(err_pred_pca, loc=params["norm_mu"], scale=params["norm_sigma"])
                + stats.t.logpdf(
                    err_est_raw,
                    df=params["t_est_raw_df"],
                    loc=params["t_est_raw_loc"],
                    scale=params["t_est_raw_scale"],
                )
                + stats.gamma.logpdf(
                    err_y,
                    a=params["gamma_k"],
                    loc=params["gamma_loc"],
                    scale=params["gamma_scale"],
                )
            )
            best_idx = int(np.argmax(loglik))
            z_star[i] = z_grid[best_idx]
            loglik_star[i] = loglik[best_idx]

            err_pred_true = z_pred[i] - z_val[i]
            err_pred_pca_true = z_pred_pca[i] - z_val[i]
            err_est_raw_true = z_est_raw[i] - z_val[i]
            err_y_true = np.linalg.norm(y_final_pred[i] - y_val[i])
            loglik_true[i] = (
                stats.t.logpdf(
                    err_pred_true,
                    df=params["t_pred_df"],
                    loc=params["t_pred_loc"],
                    scale=params["t_pred_scale"],
                )
                + stats.norm.logpdf(err_pred_pca_true, loc=params["norm_mu"], scale=params["norm_sigma"])
                + stats.t.logpdf(
                    err_est_raw_true,
                    df=params["t_est_raw_df"],
                    loc=params["t_est_raw_loc"],
                    scale=params["t_est_raw_scale"],
                )
                + stats.gamma.logpdf(
                    err_y_true,
                    a=params["gamma_k"],
                    loc=params["gamma_loc"],
                    scale=params["gamma_scale"],
                )
            )
    y_star = _y_of_z(z_star)

    y_from_z_pred = _y_of_z(z_pred)
    y_from_z_pred_pca = _y_of_z(z_pred_pca)
    y_from_z_est_raw = _y_of_z(z_est_raw)

    mee_z_pred = _mee(y_val, y_from_z_pred)
    mee_z_pred_pca = _mee(y_val, y_from_z_pred_pca)
    mee_z_est_raw = _mee(y_val, y_from_z_est_raw)
    mee_y_final = _mee(y_val, y_final_pred)
    mee_final = _mee(y_val, y_star)

    out_dir = Path("plots/errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    z_star_abs_err = np.abs(z_star - z_val)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(z_star_abs_err, bins=40, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Absolute error |z* - z|")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of |z* - z|")
    fig.tight_layout()
    hist_path = out_dir / "z_star_abs_error.png"
    fig.savefig(hist_path, dpi=150)
    print(f"Saved plot: {hist_path}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    color_vals = np.abs(z_pred - z_est_raw)
    sc = ax.scatter(loglik_true, loglik_star, s=18, alpha=0.7, c=color_vals, cmap="viridis")
    lim_min = min(loglik_true.min(), loglik_star.min())
    lim_max = max(loglik_true.max(), loglik_star.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Log-likelihood at true z")
    ax.set_ylabel("Log-likelihood at z*")
    ax.set_title("Likelihood: z* vs true z")
    fig.colorbar(sc, ax=ax, label="|z_pred - z_est_raw|")
    fig.tight_layout()
    ll_path = out_dir / "likelihood_z_star_vs_true.png"
    fig.savefig(ll_path, dpi=150)
    print(f"Saved plot: {ll_path}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    color_vals = np.abs(z_star - z_val)
    sc = ax.scatter(loglik_true, loglik_star, s=18, alpha=0.7, c=color_vals, cmap="viridis")
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Log-likelihood at true z")
    ax.set_ylabel("Log-likelihood at z*")
    ax.set_title("Likelihood: z* vs true z")
    fig.colorbar(sc, ax=ax, label="|z* - z|")
    fig.tight_layout()
    ll_path = out_dir / "likelihood_z_star_vs_true_by_z_star_err.png"
    fig.savefig(ll_path, dpi=150)
    print(f"Saved plot: {ll_path}", flush=True)

    abs_err_df = pd.DataFrame(
        {
            "z_star_abs_err": np.abs(z_star - z_val),
            "z_pred_abs_err": np.abs(z_pred - z_val),
            "z_pred_pca_abs_err": np.abs(z_pred_pca - z_val),
            "z_est_raw_abs_err": np.abs(z_est_raw - z_val),
            "y_final_pred_abs_err": np.linalg.norm(y_final_pred - y_val, axis=1),
        }
    )
    abs_err_df = abs_err_df.sort_values("z_star_abs_err", ascending=False)
    abs_err_path = out_dir / "abs_errors_sorted_by_z_star.csv"
    abs_err_df.to_csv(abs_err_path, index=False)
    print(f"Saved abs errors: {abs_err_path}", flush=True)

    out_path = Path("plots/errors/likelihood_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("MEE by estimator\n")
        f.write(f"z_pred: {mee_z_pred:.6f}\n")
        f.write(f"z_pred_pca: {mee_z_pred_pca:.6f}\n")
        f.write(f"z_est_raw: {mee_z_est_raw:.6f}\n")
        f.write(f"y_final_pred: {mee_y_final:.6f}\n")
        f.write(f"z_star (max likelihood): {mee_final:.6f}\n")
    print(f"Saved results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
