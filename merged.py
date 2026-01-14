import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import differential_evolution


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
FIT_PATH = "fit.txt"

N_COMPONENTS = 6
N_PCA_COMPONENTS = 5
K_NEIGHBORS = 2
RIDGE_ALPHA = 0.0000001
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]
GAMMA = 0.4
PC_INTERCEPT = -7.4426
PC_BETAS = np.array([-10.9003, 2.6545])
LOCAL_PCA_COMPONENTS = 4
LOCAL_RIDGE_ALPHA = 0.01
BETA = 5.0
BETA_SWEEP = np.linspace(0.0, 10.0, 21)


def _parse_fit(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    T = None
    const = 0.0
    terms = []
    for line in lines:
        if line.startswith("T ="):
            T = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("f(") or line.startswith("x ="):
            continue
        if re.fullmatch(r"[+-]?[0-9.eE+-]+", line):
            const += float(line)
            continue
        m = re.match(r"([+-])\s*([0-9.eE+-]+)\s*\*\s*(sin|cos)\((\d+)\*pi\*x/T\)", line)
        if not m:
            raise ValueError(f"Unrecognized line in fit: {line}")
        sign, coeff, kind, k = m.groups()
        coeff = float(coeff)
        if sign == "-":
            coeff = -coeff
        terms.append((coeff, kind, int(k)))
    if T is None:
        raise ValueError("Missing T in fit file")
    return T, const, terms


def _fit_eval(z, T, const, terms):
    out = const
    for coeff, kind, k in terms:
        arg = (k * np.pi * z) / T
        out += coeff * (np.sin(arg) if kind == "sin" else np.cos(arg))
    return out


def _fit_deriv(z, T, terms):
    out = 0.0
    for coeff, kind, k in terms:
        arg = (k * np.pi * z) / T
        factor = (k * np.pi) / T
        if kind == "sin":
            out += coeff * np.cos(arg) * factor
        else:
            out += -coeff * np.sin(arg) * factor
    return out


def _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam):
    grid = z0 + np.linspace(-T / 2.0, T / 2.0, 41)
    grid = np.clip(grid, z_min, z_max)
    vals = _fit_eval(grid, T, const, terms)
    obj = (vals - pc2) ** 2 + lam * (grid - z0) ** 2
    z = grid[np.argmin(obj)]

    for _ in range(30):
        fz = _fit_eval(z, T, const, terms)
        dfz = _fit_deriv(z, T, terms)
        grad = 2 * (fz - pc2) * dfz + 2 * lam * (z - z0)
        hess = 2 * (dfz ** 2) + 2 * lam
        if abs(hess) < 1e-8:
            break
        step = grad / hess
        if abs(step) > 5.0:
            step = np.sign(step) * 5.0
        z_new = np.clip(z - step, z_min, z_max)
        if abs(z_new - z) < 1e-6:
            z = z_new
            break
        z = z_new
    return z


def _reconstruct_y(z):
    k1 = 0.5463
    k2 = 1.1395
    y1_rec = k1 * z * np.cos(k2 * z)
    y2_rec = k1 * z * np.sin(k2 * z)
    sum_rec = -z * np.cos(2 * z)
    y3_rec = (sum_rec + z) / 2.0
    y4_rec = (sum_rec - z) / 2.0
    return np.column_stack([y1_rec, y2_rec, y3_rec, y4_rec])


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


def _d_perp(y_hat, R=80.0, n_grid=4001):
    y_hat = np.asarray(y_hat)
    z0 = y_hat[2] - y_hat[3]
    zs = np.linspace(z0 - R, z0 + R, n_grid)
    ys = _y_of_z(zs)
    dists = np.linalg.norm(ys - y_hat, axis=1)
    return float(np.min(dists))


def _calculate_mee(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))


def _local_ridge_predict(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train):
    W_sqrt = np.sqrt(np.maximum(weights, 0))
    X_train_weighted = X_train_pca * W_sqrt
    X_val_weighted = X_val_pca * W_sqrt

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree").fit(X_train_weighted)
    _, indices = nbrs.kneighbors(X_val_weighted)

    y_preds = []
    for i in range(len(X_val_sc)):
        idx = indices[i]
        X_loc = X_train_sc[idx]
        Y_loc = Y_train[idx]
        model = Ridge(alpha=LOCAL_RIDGE_ALPHA)
        model.fit(X_loc, Y_loc)
        pred = model.predict(X_val_sc[i:i + 1])
        y_preds.append(pred[0])

    return np.array(y_preds)


def _objective_function(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train, Y_val):
    y_pred = _local_ridge_predict(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train)
    return _calculate_mee(Y_val, y_pred)


def _scatter_plot(x, y, out_path, xlabel, ylabel, title, color, diagonal=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=18, alpha=0.7, color=color)
    if diagonal:
        x_line, y_line = diagonal
        ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)


def main(do_plots=True):
    df = pd.read_csv(FILE_PATH, comment="#", header=None, index_col=0)
    X = df.iloc[:, 0:12].values
    Y = df.iloc[:, 12:16].values
    z = Y[:, 2] - Y[:, 3]

    random_state = int(time.time_ns() % 1_000_000)
    X_train, X_val, Y_train, Y_val, z_train, z_val = train_test_split(
        X, Y, z, test_size=0.1, random_state=random_state
    )
    print(f"random_state: {random_state}", flush=True)

    # Shared scaler + PCA (fit once, reuse slices)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    max_components = max(N_COMPONENTS, N_PCA_COMPONENTS, 2)
    pca = PCA(n_components=max_components)
    X_train_pca_full = pca.fit_transform(X_train_scaled)
    X_val_pca_full = pca.transform(X_val_scaled)

    X_train_pca = X_train_pca_full[:, :N_COMPONENTS]
    X_val_pca = X_val_pca_full[:, :N_COMPONENTS]

    # Local ridge optimization (from test3.py) to get y_final_pred
    X_train_pca_local = X_train_pca_full[:, :LOCAL_PCA_COMPONENTS]
    X_val_pca_local = X_val_pca_full[:, :LOCAL_PCA_COMPONENTS]
    bounds = [(0, 10)] * LOCAL_PCA_COMPONENTS
    result = differential_evolution(
        _objective_function,
        bounds,
        args=(X_train_scaled, X_val_scaled, X_train_pca_local, X_val_pca_local, Y_train, Y_val),
        strategy="best1bin",
        maxiter=10,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True,
    )
    best_weights = result.x
    y_final_pred = _local_ridge_predict(
        best_weights, X_train_scaled, X_val_scaled, X_train_pca_local, X_val_pca_local, Y_train
    )
    mee_de = _calculate_mee(Y_val, y_final_pred)
    print(f"MEE differential evolution: {mee_de:.5f}", flush=True)

    model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
    model_rf.fit(z_train.reshape(-1, 1), X_train_pca)

    z_min, z_max = z_train.min() - 10, z_train.max() + 10
    z_grid = np.linspace(z_min, z_max, 500000)
    X_grid_pca_pred = model_rf.predict(z_grid.reshape(-1, 1))

    def predict_z_with_pca(input_raw):
        x_scaled = scaler.transform([input_raw])
        x_pca = pca.transform(x_scaled)[:, :N_COMPONENTS]
        dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
        return z_grid[np.argmin(dists)]

    z_est = np.array([predict_z_with_pca(x) for x in X_val])

    # KNN z_pred (no refinement)
    X_train_sc = X_train_scaled
    X_val_sc = X_val_scaled

    X_train_pca_knn = X_train_pca_full[:, :N_PCA_COMPONENTS]
    X_val_pca_knn = X_val_pca_full[:, :N_PCA_COMPONENTS]

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree").fit(X_train_pca_knn)
    _, indices = nbrs.kneighbors(X_val_pca_knn)

    z_pred_val = []
    for i in range(len(X_val)):
        idx = indices[i]
        X_loc = X_train_sc[idx]
        z_loc = z_train[idx]
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_loc, z_loc)
        pred = model.predict([X_val_sc[i]])[0]
        z_pred_val.append(pred)

    z_pred_val = np.array(z_pred_val).flatten()

    # pc1/pc2 for linear PC model + pc2 for refinement
    pc_val = X_val_pca_full[:, :2]
    pc2_val = pc_val[:, 1]
    z_pred_pc = PC_INTERCEPT + pc_val @ PC_BETAS

    T, const, terms = _parse_fit(FIT_PATH)
    z_min = float(z.min())
    z_max = float(z.max())

    # Refine z_pred_val -> z_ref
    best = None
    for lam in LAMBDA_SWEEP:
        z_ref = np.array(
            [
                _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                for z0, pc2 in zip(z_pred_val, pc2_val)
            ]
        )
        Y_pred = _reconstruct_y(z_ref)
        mee = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))
        if best is None or mee < best[0]:
            best = (mee, lam, z_ref)

    best_mee, best_lam, z_ref = best

    # Refine z_est before ensemble
    z_est_raw = z_est
    best = None
    for lam in LAMBDA_SWEEP:
        z_est_ref = np.array(
            [
                _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                for z0, pc2 in zip(z_est_raw, pc2_val)
            ]
        )
        Y_pred = _reconstruct_y(z_est_ref)
        mee = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))
        if best is None or mee < best[0]:
            best = (mee, lam, z_est_ref)

    best_mee_est, best_lam_est, z_est = best
    print(f"Best lambda (MEE) for z_est: {best_lam_est} -> MEE: {best_mee_est:.5f}", flush=True)
    print(f"Best lambda (MEE) for z_ref: {best_lam} -> MEE: {best_mee:.5f}", flush=True)

    # MEE for individual methods
    Y_est = _reconstruct_y(z_est)
    Y_ref = _reconstruct_y(z_ref)
    mee_est = np.mean(np.linalg.norm(Y_est - Y_val, axis=1))
    mee_ref = np.mean(np.linalg.norm(Y_ref - Y_val, axis=1))
    print(f"MEE z_est: {mee_est:.5f}", flush=True)
    print(f"MEE z_ref: {mee_ref:.5f}", flush=True)

    # Ensemble rule: average if close, else random pick
    z_ens = np.empty_like(z_est)
    close_mask = np.abs(z_ref - z_est) < GAMMA
    z_ens[close_mask] = 0.5 * (z_ref[close_mask] + z_est[close_mask])
    far_idx = np.where(~close_mask)[0]
    print(f"Points with |z_ref - z_est| >= GAMMA: {far_idx.size}", flush=True)
    if far_idx.size:
        dist_ref = np.abs(z_ref[far_idx] - z_pred_pc[far_idx])
        dist_est = np.abs(z_est[far_idx] - z_pred_pc[far_idx])
        pick_ref = dist_ref <= dist_est
        z_ens[far_idx] = np.where(pick_ref, z_ref[far_idx], z_est[far_idx])

    Y_ens = _reconstruct_y(z_ens)
    mee_ens = np.mean(np.linalg.norm(Y_ens - Y_val, axis=1))
    print(f"MEE ensemble (pre-refine): {mee_ens:.5f}", flush=True)

    # Refinement on ensemble z
    best = None
    for lam in LAMBDA_SWEEP:
        z_refined = np.array(
            [
                _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                for z0, pc2 in zip(z_ens, pc2_val)
            ]
        )
        Y_pred = _reconstruct_y(z_refined)
        mee = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))
        if best is None or mee < best[0]:
            best = (mee, lam, z_refined)

    best_mee_ens, best_lam_ens, z_refined_ens = best
    print(f"Ensemble refine best lambda (MEE): {best_lam_ens} -> MEE: {best_mee_ens:.5f}", flush=True)

    # Select y_final_pred if d_perp(y_final_pred) < BETA, else y(z_refined_ens)
    y_refined_ens = _reconstruct_y(z_refined_ens)
    d_perp_vals = np.array([_d_perp(y_hat) for y_hat in y_final_pred])

    # Select optimal beta on the current split
    betas = np.asarray(BETA_SWEEP)
    mee_by_beta = []
    for beta in betas:
        mask = d_perp_vals < beta
        y_beta = np.where(mask[:, None], y_final_pred, y_refined_ens)
        mee_by_beta.append(np.mean(np.linalg.norm(y_beta - Y_val, axis=1)))
    best_idx = int(np.argmin(mee_by_beta))
    beta_opt = float(betas[best_idx])
    print(f"Best beta (current split): {beta_opt}", flush=True)

    use_final = d_perp_vals < beta_opt
    y_selected = np.where(use_final[:, None], y_final_pred, y_refined_ens)
    mee_selected = np.mean(np.linalg.norm(y_selected - Y_val, axis=1))
    print(f"MEE y_selected (d_perp < {beta_opt}): {mee_selected:.5f}", flush=True)

    metrics = {
        "mee_de": mee_de,
        "mee_z_est": mee_est,
        "mee_z_ref": mee_ref,
        "mee_ens_refined": best_mee_ens,
        "mee_selected": mee_selected,
    }
    if not do_plots:
        return metrics

    out_path = Path("plots/mee_vs_beta.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(betas, mee_by_beta, color="tab:blue", linewidth=2)
    ax.set_xlabel("beta")
    ax.set_ylabel("MEE")
    ax.set_title("MEE vs beta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)

    # Histogram of absolute errors for z_refined_ens
    err_refined_ens = np.abs(z_refined_ens - z_val)
    out_path = Path("plots/z_refined_ens_abs_error_hist.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(err_refined_ens, bins=40, color="tab:green", alpha=0.8)
    ax.set_xlabel("Absolute error |z_refined_ens - z_val|")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of absolute errors (z_refined_ens)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)

    # Plot abs error: z_est vs z_ref
    err_est = np.abs(z_est - z_val)
    err_ref = np.abs(z_ref - z_val)

    out_path = Path("plots/z_est_vs_z_ref_abs_error.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lim = max(err_est.max(), err_ref.max())
    _scatter_plot(
        err_est,
        err_ref,
        out_path,
        "Absolute error |z_est - z_val|",
        "Absolute error |z_ref - z_val|",
        "z_est error vs z_ref error",
        "tab:blue",
        diagonal=([0, lim], [0, lim]),
    )

    # Plot signed error: z_est vs z_ref
    err_est_signed = z_est - z_val
    err_ref_signed = z_ref - z_val

    out_path = Path("plots/z_est_vs_z_ref_signed_error.png")
    lim = max(np.max(np.abs(err_est_signed)), np.max(np.abs(err_ref_signed)))
    _scatter_plot(
        err_est_signed,
        err_ref_signed,
        out_path,
        "Signed error (z_est - z_val)",
        "Signed error (z_ref - z_val)",
        "z_est signed error vs z_ref signed error",
        "tab:purple",
        diagonal=([-lim, lim], [-lim, lim]),
    )

    # Plot abs error z_pred (PC) vs max(abs err z_ref, abs err z_est)
    err_pred_pc = np.abs(z_pred_pc - z_val)
    err_max = np.maximum(err_ref, err_est)

    out_path = Path("plots/z_pred_pc_vs_max_err.png")
    lim = max(err_pred_pc.max(), err_max.max())
    _scatter_plot(
        err_pred_pc,
        err_max,
        out_path,
        "Absolute error |z_pred_pc - z_val|",
        "max(|z_ref - z_val|, |z_est - z_val|)",
        "z_pred_pc error vs max(z_ref, z_est) error",
        "tab:cyan",
        diagonal=([0, lim], [0, lim]),
    )

    # Plot |z_ref - z_est| vs max(|z_ref - z_val|, |z_est - z_val|)
    err_gap = np.abs(z_ref - z_est)
    err_max = np.maximum(err_ref, err_est)
    out_path = Path("plots/z_ref_gap_vs_max_err.png")
    lim = max(err_gap.max(), err_max.max())
    _scatter_plot(
        err_gap,
        err_max,
        out_path,
        "|z_ref - z_est|",
        "max(|z_ref - z_val|, |z_est - z_val|)",
        "|z_ref - z_est| vs max error",
        "tab:green",
        diagonal=([0, lim], [0, lim]),
    )

    # Plot y_final_pred error vs z_refined_ens error
    y_final_err = np.linalg.norm(y_final_pred - Y_val, axis=1)
    z_refined_err = np.abs(z_refined_ens - z_val)
    y_final_err_norm = y_final_err / max(y_final_err.max(), 1e-12)
    out_path = Path("plots/y_final_pred_err_vs_z_refined_ens_err.png")
    lim_y = z_refined_err.max()
    _scatter_plot(
        y_final_err_norm,
        z_refined_err,
        out_path,
        "Normalized y_final_pred error (MEE per sample)",
        "Absolute error |z_refined_ens - z_val|",
        "Normalized y_final_pred error vs z_refined_ens error",
        "tab:blue",
        diagonal=([0, 1], [0, lim_y]),
    )

    # Plot MEE(y_final_pred) vs MEE(z_refined_ens) without normalization
    y_refined_ens = _reconstruct_y(z_refined_ens)
    y_refined_err = np.linalg.norm(y_refined_ens - Y_val, axis=1)
    out_path = Path("plots/y_final_pred_mee_vs_z_refined_ens_mee.png")
    lim = max(y_final_err.max(), y_refined_err.max())
    y_dist = np.linalg.norm(y_final_pred - y_refined_ens, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(y_final_err, y_refined_err, s=18, alpha=0.7, c=y_dist, cmap="viridis")
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("MEE per sample (y_final_pred)")
    ax.set_ylabel("MEE per sample (z_refined_ens)")
    ax.set_title("MEE: y_final_pred vs z_refined_ens")
    fig.colorbar(sc, ax=ax, label="|y_final_pred - y_refined_ens|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)


if __name__ == "__main__":
    main()
