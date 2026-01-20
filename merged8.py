import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
FIT_PATH = "fit.txt"
N_COMPONENTS = 6
LAMBDA_FIXED = 0.1
GRID_STEPS = 201


def _reconstruct_y_from_z(z_val):
    y1 = 0.5463 * z_val * np.cos(1.1395 * z_val)
    y2 = 0.5463 * z_val * np.sin(1.1395 * z_val)
    sum_y34 = -z_val * np.cos(2 * z_val)
    diff_y34 = z_val
    y3 = (sum_y34 + diff_y34) / 2
    y4 = (sum_y34 - diff_y34) / 2
    return np.array([y1, y2, y3, y4])


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


def _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam):
    grid = z0 + np.linspace(-T / 2.0, T / 2.0, GRID_STEPS)
    grid = np.clip(grid, z_min, z_max)
    vals = _fit_eval(grid, T, const, terms)
    obj = (vals - pc2) ** 2 + lam * (grid - z0) ** 2
    return grid[int(np.argmin(obj))]


def main():
    df = pd.read_csv(FILE_PATH, comment="#", header=None, index_col=0)
    X = df.iloc[:, 0:12].values
    Y = df.iloc[:, 12:16].values
    z = Y[:, 2] - Y[:, 3]

    T, const, terms = _parse_fit(FIT_PATH)

    random_state = int(time.time_ns() % 1_000_000)
    print(f"random_state: {random_state}", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    all_errors = []
    all_errors_nn = []
    all_errors_refined = []
    all_z_errors = []
    all_z_vals = []

    out_dir = Path("plots/8")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        z_train, z_val = z[train_idx], z[val_idx]

        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)

        pca = PCA(n_components=N_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        pc2_val = X_val_pca[:, 1]

        print(f"--- SETUP PCA (fold {fold_idx}) ---")
        print(f"Componenti utilizzate: {N_COMPONENTS}")
        print(f"Varianza Spiegata Cumulativa: {np.sum(pca.explained_variance_ratio_):.4%}")

        print("\nAddestramento Random Forest (z -> PCA features)...")
        model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
        model_rf.fit(z_train.reshape(-1, 1), X_train_pca)

        z_min, z_max = z_train.min() - 10, z_train.max() + 10
        z_grid = np.linspace(z_min, z_max, 500000)
        X_grid_pca_pred = model_rf.predict(z_grid.reshape(-1, 1))
        X_grid_pca_pred[:, 1] = _fit_eval(z_grid, T, const, terms)

        def predict_z_with_pca(input_raw):
            x_scaled = scaler_X.transform([input_raw])
            x_pca = pca.transform(x_scaled)[0]
            dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
            best_idx = np.argmin(dists)
            diff_best = X_grid_pca_pred[best_idx] - x_pca
            return z_grid[best_idx], diff_best

        print("\nValutazione MEE sul Validation Set...")
        errors = []
        z_errors = []

        z_random_forest_val = []
        diff_best_val = []
        for i in range(len(X_val)):
            z_rf, diff_best = predict_z_with_pca(X_val[i])
            z_random_forest_val.append(z_rf)
            diff_best_val.append(diff_best)
        z_random_forest_val = np.array(z_random_forest_val)

        errors_nn = []
        for i in range(len(X_val)):
            y_est = _reconstruct_y_from_z(z_random_forest_val[i])
            err = np.linalg.norm(y_est - Y_val[i])
            errors_nn.append(err)
        mee_nn = np.mean(errors_nn)
        print(f"MEE Fold {fold_idx} (closest, no refine): {mee_nn:.5f}")
        all_errors_nn.extend(errors_nn)

        z_min_ref = float(z_train.min())
        z_max_ref = float(z_train.max())
        z_random_forest_val_refined = np.array(
            [
                _refine_z_reg(z0, pc2, T, const, terms, z_min_ref, z_max_ref, LAMBDA_FIXED)
                for z0, pc2 in zip(z_random_forest_val, pc2_val)
            ]
        )

        z_errors = np.abs(z_random_forest_val_refined - z_val)
        for i in range(len(X_val)):
            y_est = _reconstruct_y_from_z(z_random_forest_val_refined[i])
            err = np.linalg.norm(y_est - Y_val[i])
            errors.append(err)
        mee_refined = np.mean(errors)
        print(f"MEE Fold {fold_idx} (closest, refined): {mee_refined:.5f}")
        all_errors_refined.extend(errors)

        mee = np.mean(errors)
        print(f"MEE Fold {fold_idx}: {mee:.5f}")
        all_errors.extend(errors)
        all_z_errors.extend(z_errors)
        all_z_vals.extend(z_val)

        diff_best_val = np.asarray(diff_best_val)
        out_path = out_dir / f"pca_diff_comp1_vs_comp2_fold_{fold_idx}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            diff_best_val[:, 0],
            diff_best_val[:, 1],
            s=18,
            alpha=0.7,
            c=z_errors,
            cmap="viridis",
        )
        ax.set_xlabel("Component 1 of (X_grid_pca_pred - x_pca)")
        ax.set_ylabel("Component 2 of (X_grid_pca_pred - x_pca)")
        ax.set_title(f"Diff components (fold {fold_idx})")
        fig.colorbar(sc, ax=ax, label="Absolute error |z_random_forest - z_val|")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}", flush=True)

    mee = np.mean(all_errors)
    print(f"MEE Totale (CV): {mee:.5f}")
    if all_errors_nn:
        mee_nn_all = np.mean(all_errors_nn)
        print(f"MEE Totale (closest, no refine): {mee_nn_all:.5f}")
    if all_errors_refined:
        mee_refined_all = np.mean(all_errors_refined)
        print(f"MEE Totale (closest, refined): {mee_refined_all:.5f}")

    all_z_errors_arr = np.asarray(all_z_errors)
    all_z_vals_arr = np.asarray(all_z_vals)

    out_path = out_dir / "z_random_forest_abs_error_hist.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_z_errors_arr, bins=40, color="tab:orange", alpha=0.8)
    ax.set_xlabel("Absolute error |z_random_forest - z_val|")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of absolute errors (z_random_forest)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)

    out_path = out_dir / "z_random_forest_abs_error_vs_z_val.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_vals_arr, all_z_errors_arr, s=18, alpha=0.7, color="tab:blue")
    ax.set_xlabel("z_val")
    ax.set_ylabel("Absolute error |z_random_forest - z_val|")
    ax.set_title("Absolute error vs z_val (z_random_forest)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)


if __name__ == "__main__":
    main()
