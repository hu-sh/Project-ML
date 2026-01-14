import re
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


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
FIT_PATH = "fit.txt"

N_COMPONENTS = 6
N_PCA_COMPONENTS = 5
K_NEIGHBORS = 2
RIDGE_ALPHA = 0.0000001
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]
GAMMA = 1.0


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


def main():
    df = pd.read_csv(FILE_PATH, comment="#", header=None, index_col=0)
    X = df.iloc[:, 0:12].values
    Y = df.iloc[:, 12:16].values
    z = Y[:, 2] - Y[:, 3]

    X_train, X_val, Y_train, Y_val, z_train, z_val = train_test_split(
        X, Y, z, test_size=0.1, random_state=42
    )

    # Test9-style z_est
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
    model_rf.fit(z_train.reshape(-1, 1), X_train_pca)

    z_min, z_max = z_train.min() - 10, z_train.max() + 10
    z_grid = np.linspace(z_min, z_max, 500000)
    X_grid_pca_pred = model_rf.predict(z_grid.reshape(-1, 1))

    def predict_z_with_pca(input_raw):
        x_scaled = scaler_X.transform([input_raw])
        x_pca = pca.transform(x_scaled)
        dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
        return z_grid[np.argmin(dists)]

    z_est = np.array([predict_z_with_pca(x) for x in X_val])

    # KNN z_pred (no refinement)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    pca_knn = PCA(n_components=N_PCA_COMPONENTS)
    X_train_pca_knn = pca_knn.fit_transform(X_train_sc)
    X_val_pca_knn = pca_knn.transform(X_val_sc)

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

    # MEE for individual methods
    Y_est = _reconstruct_y(z_est)
    Y_pred = _reconstruct_y(z_pred_val)
    mee_est = np.mean(np.linalg.norm(Y_est - Y_val, axis=1))
    mee_pred = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))
    print(f"MEE z_est: {mee_est:.5f}")
    print(f"MEE z_pred_val: {mee_pred:.5f}")

    # pc2 for refinement
    pca2 = PCA(n_components=2, random_state=0)
    X_train_pca2 = pca2.fit_transform(X_train_sc)
    X_val_pca2 = pca2.transform(X_val_sc)
    pc2_val = X_val_pca2[:, 1]

    T, const, terms = _parse_fit(FIT_PATH)
    z_min = float(z.min())
    z_max = float(z.max())

    # Ensemble rule: average if close, else random pick
    rng = np.random.default_rng()
    z_ens = np.empty_like(z_est)
    close_mask = np.abs(z_pred_val - z_est) < GAMMA
    z_ens[close_mask] = 0.5 * (z_pred_val[close_mask] + z_est[close_mask])
    far_idx = np.where(~close_mask)[0]
    if far_idx.size:
        pick_pred = rng.random(far_idx.size) < 0.5
        z_ens[far_idx] = np.where(pick_pred, z_pred_val[far_idx], z_est[far_idx])

    Y_ens = _reconstruct_y(z_ens)
    mee_ens = np.mean(np.linalg.norm(Y_ens - Y_val, axis=1))
    print(f"MEE ensemble (pre-refine): {mee_ens:.5f}")

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
    print(f"Ensemble refine best lambda (MEE): {best_lam_ens} -> MEE: {best_mee_ens:.5f}")

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
    print(f"Saved plot: {out_path}")

    # Plot abs error: z_est vs z_pred_val
    err_est = np.abs(z_est - z_val)
    err_pred = np.abs(z_pred_val - z_val)

    out_path = Path("plots/z_est_vs_z_pred_abs_error.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(err_est, err_pred, s=18, alpha=0.7, color="tab:blue")
    lim = max(err_est.max(), err_pred.max())
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute error |z_est - z_val|")
    ax.set_ylabel("Absolute error |z_pred_val - z_val|")
    ax.set_title("z_est error vs z_pred_val error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    # Plot signed error: z_est vs z_pred_val
    err_est_signed = z_est - z_val
    err_pred_signed = z_pred_val - z_val

    out_path = Path("plots/z_est_vs_z_pred_signed_error.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(err_est_signed, err_pred_signed, s=18, alpha=0.7, color="tab:purple")
    lim = max(np.max(np.abs(err_est_signed)), np.max(np.abs(err_pred_signed)))
    ax.plot([-lim, lim], [-lim, lim], color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Signed error (z_est - z_val)")
    ax.set_ylabel("Signed error (z_pred_val - z_val)")
    ax.set_title("z_est signed error vs z_pred_val signed error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
