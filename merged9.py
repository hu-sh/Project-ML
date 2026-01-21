import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
FIT_PATH = "fit.txt"
W_THEORETICAL = (1 + 1 / np.sqrt(2)) / 3
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]
GRID_STEPS = 201
GRID_SIZE = 5000
K_SSD = 100
N_COMPONENTS = 6


def get_physical_features(z_arr):
    if np.isscalar(z_arr):
        z_arr = np.array([z_arr])
    feats = [z_arr]
    for k in range(1, 10):
        feats.append(np.sin(k * W_THEORETICAL * z_arr))
        feats.append(np.cos(k * W_THEORETICAL * z_arr))
    return np.column_stack(feats)


def reconstruct_targets(z_in):
    z = np.array(z_in)
    y1 = 0.5463 * z * np.cos(1.1395 * z)
    y2 = 0.5463 * z * np.sin(1.1395 * z)
    sum_y3y4 = -z * np.cos(2 * z)
    diff_y3y4 = z
    y3 = (sum_y3y4 + diff_y3y4) / 2.0
    y4 = (sum_y3y4 - diff_y3y4) / 2.0
    return np.column_stack([y1, y2, y3, y4])


def calculate_mee(y_true, y_pred):
    errors = y_true - y_pred
    dist = np.sqrt(np.sum(errors ** 2, axis=1))
    return np.mean(dist)


def parse_fit(path):
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


def fit_eval(z, T, const, terms):
    out = const
    for coeff, kind, k in terms:
        arg = (k * np.pi * z) / T
        out += coeff * (np.sin(arg) if kind == "sin" else np.cos(arg))
    return out


def refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam):
    grid = z0 + np.linspace(-T / 2.0, T / 2.0, GRID_STEPS)
    grid = np.clip(grid, z_min, z_max)
    vals = fit_eval(grid, T, const, terms)
    obj = (vals - pc2) ** 2 + lam * (grid - z0) ** 2
    return grid[int(np.argmin(obj))]


def main():
    df = pd.read_csv(FILE_PATH, comment="#", header=None)
    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values
    z_all = Y[:, 2] - Y[:, 3]

    T, const, terms = parse_fit(FIT_PATH)

    random_state = 45
    print(f"random_state: {random_state}")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    out_dir = Path("plots/9")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_z_abs_err_before = []
    all_z_abs_err_refined = []
    all_z_abs_err = []
    all_z_abs_err_forest = []
    all_z_abs_err_forest_refined = []
    all_z_test5 = []
    all_z_forest_refined = []
    all_y_val = []
    all_y_abs_err = []
    all_weighted_ssd = []
    all_weighted_ssd_true = []
    all_unweighted_ssd = []
    all_ssd_ratio = []
    all_z_var = []
    all_z_refine_delta = []
    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        z_train, z_val = z_all[train_idx], z_all[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        print(f"\nFold {fold_idx}:")

        # --- test5 pipeline (harmonic forward model + inverse search) ---
        models = []
        weights = []
        train_feats = get_physical_features(z_train)
        for i in range(12):
            reg = LinearRegression().fit(train_feats, X_train[:, i])
            models.append(reg)
            preds = reg.predict(train_feats)
            mse = np.mean((X_train[:, i] - preds) ** 2)
            w = 1.0 / mse if mse > 1e-9 else 1.0
            weights.append(w)
        weights = np.array(weights)

        coefs_mat = np.array([m.coef_ for m in models])
        intercepts = np.array([m.intercept_ for m in models])

        z_min, z_max = z_train.min() - 5, z_train.max() + 5
        z_grid = np.linspace(z_min, z_max, GRID_SIZE)
        grid_feats = get_physical_features(z_grid)
        grid_preds = grid_feats @ coefs_mat.T + intercepts

        z_recovered = []
        z_candidates_all = []

        for i in range(len(X_val)):
            x_obs = X_val[i]
            diff_sq = (grid_preds - x_obs) ** 2
            weighted_ssd = np.sum(diff_sq * weights, axis=1)
            unweighted_ssd = np.sum(diff_sq, axis=1)

            best_grid_idx = np.argmin(weighted_ssd)
            z_init = z_grid[best_grid_idx]
            all_weighted_ssd.append(float(weighted_ssd[best_grid_idx]))
            all_unweighted_ssd.append(float(unweighted_ssd[best_grid_idx]))
            denom = float(weighted_ssd[best_grid_idx])
            all_ssd_ratio.append(float(unweighted_ssd[best_grid_idx]) / max(denom, 1e-12))

            k = min(K_SSD, len(weighted_ssd))
            idx_k = np.argpartition(weighted_ssd, k - 1)[:k]
            z_candidates = z_grid[idx_k]
            all_z_var.append(float(np.var(z_candidates)))
            z_candidates_all.append(z_candidates)

            z_recovered.append(z_init)
            f_true = get_physical_features(z_val[i]).flatten()
            x_true_pred = coefs_mat.dot(f_true) + intercepts
            all_weighted_ssd_true.append(float(np.sum(((x_true_pred - x_obs) ** 2) * weights)))

        z_recovered = np.array(z_recovered)

        # PCA for pc2 used by refinement
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        pca_small = PCA(n_components=2)
        X_train_pca = pca_small.fit_transform(X_train_sc)
        X_val_pca = pca_small.transform(X_val_sc)
        pc2_train = X_train_pca[:, 1]
        pc2_val = X_val_pca[:, 1]

        z_train_init = []
        for i in range(len(X_train)):
            x_obs = X_train[i]
            diff_sq = (grid_preds - x_obs) ** 2
            weighted_ssd = np.sum(diff_sq * weights, axis=1)
            z_train_init.append(z_grid[int(np.argmin(weighted_ssd))])
        z_train_init = np.array(z_train_init)

        z_min_ref = float(z_train.min())
        z_max_ref = float(z_train.max())
        best = None
        for lam in LAMBDA_SWEEP:
            z_ref = np.array(
                [
                    refine_z_reg(z0, pc2, T, const, terms, z_min_ref, z_max_ref, lam)
                    for z0, pc2 in zip(z_train_init, pc2_train)
                ]
            )
            Y_pred_train = reconstruct_targets(z_ref)
            mee_train = calculate_mee(Y_train, Y_pred_train)
            if best is None or mee_train < best[0]:
                best = (mee_train, lam)
        best_lam = best[1]

        z_recovered_refined = np.array(
            [
                refine_z_reg(z0, pc2, T, const, terms, z_min_ref, z_max_ref, best_lam)
                for z0, pc2 in zip(z_recovered, pc2_val)
            ]
        )
        all_z_refine_delta.extend(np.abs(z_recovered_refined - z_recovered))

        z_recovered_refined_snapped = []
        for i in range(len(z_recovered_refined)):
            candidates = z_candidates_all[i]
            z_ref = z_recovered_refined[i]
            has_lower = np.any(candidates < z_ref)
            has_upper = np.any(candidates > z_ref)
            if has_lower and has_upper:
                snapped = candidates[np.argmin(np.abs(candidates - z_ref))]
            else:
                snapped = z_ref
            z_recovered_refined_snapped.append(snapped)
        z_recovered_refined_snapped = np.array(z_recovered_refined_snapped)

        all_z_abs_err_before.extend(np.abs(z_recovered - z_val))
        all_z_abs_err_refined.extend(np.abs(z_recovered_refined - z_val))
        all_z_abs_err.extend(np.abs(z_recovered_refined_snapped - z_val))
        Y_pred_snapped = reconstruct_targets(z_recovered_refined_snapped)
        all_y_abs_err.extend(np.linalg.norm(Y_pred_snapped - Y_val, axis=1))
        y_test5_before = reconstruct_targets(z_recovered)
        y_test5_refined = reconstruct_targets(z_recovered_refined)
        y_test5_snapped = Y_pred_snapped

        # --- test9 pipeline (Random Forest + PCA inversion) ---
        scaler_rf = StandardScaler()
        X_train_scaled = scaler_rf.fit_transform(X_train)
        X_val_scaled = scaler_rf.transform(X_val)

        pca_rf = PCA(n_components=N_COMPONENTS)
        X_train_pca_rf = pca_rf.fit_transform(X_train_scaled)
        X_val_pca_rf = pca_rf.transform(X_val_scaled)
        pc2_train_rf = X_train_pca_rf[:, 1]
        pc2_val_rf = X_val_pca_rf[:, 1]

        model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
        model_rf.fit(z_train.reshape(-1, 1), X_train_pca_rf)

        z_min_rf, z_max_rf = z_train.min() - 10, z_train.max() + 10
        z_grid_rf = np.linspace(z_min_rf, z_max_rf, 500000)
        X_grid_pca_pred = model_rf.predict(z_grid_rf.reshape(-1, 1))

        z_forest_train = []
        for i in range(len(X_train)):
            x_scaled = scaler_rf.transform([X_train[i]])
            x_pca = pca_rf.transform(x_scaled)[0]
            dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
            z_forest_train.append(z_grid_rf[int(np.argmin(dists))])
        z_forest_train = np.array(z_forest_train)

        best_lam_forest = None
        for lam in LAMBDA_SWEEP:
            z_ref = np.array(
                [
                    refine_z_reg(z0, pc2, T, const, terms, z_min_rf, z_max_rf, lam)
                    for z0, pc2 in zip(z_forest_train, pc2_train_rf)
                ]
            )
            Y_pred_train = reconstruct_targets(z_ref)
            mee_train = calculate_mee(Y_train, Y_pred_train)
            if best_lam_forest is None or mee_train < best_lam_forest[0]:
                best_lam_forest = (mee_train, lam)
        best_lam_forest = best_lam_forest[1]

        z_forest = []
        for i in range(len(X_val)):
            x_scaled = scaler_rf.transform([X_val[i]])
            x_pca = pca_rf.transform(x_scaled)[0]
            dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
            z_forest.append(z_grid_rf[int(np.argmin(dists))])
        z_forest = np.array(z_forest)
        z_forest_refined = np.array(
            [
                refine_z_reg(z0, pc2, T, const, terms, z_min_rf, z_max_rf, best_lam_forest)
                for z0, pc2 in zip(z_forest, pc2_val_rf)
            ]
        )
        y_forest = reconstruct_targets(z_forest)
        y_forest_refined = reconstruct_targets(z_forest_refined)
        all_z_abs_err_forest.extend(np.abs(z_forest - z_val))
        all_z_abs_err_forest_refined.extend(np.abs(z_forest_refined - z_val))
        all_z_test5.extend(z_recovered_refined_snapped)
        all_z_forest_refined.extend(z_forest_refined)
        all_y_val.append(Y_val)

        for row_idx, sample_idx in enumerate(val_idx):
            rows.append(
                {
                    "sample_id": int(sample_idx),
                    "z_true": float(z_val[row_idx]),
                    "z_test5_before": float(z_recovered[row_idx]),
                    "z_test5_refined": float(z_recovered_refined[row_idx]),
                    "z_test5_snapped": float(z_recovered_refined_snapped[row_idx]),
                    "z_forest": float(z_forest[row_idx]),
                    "z_forest_refined": float(z_forest_refined[row_idx]),
                    "y_true_1": float(Y_val[row_idx, 0]),
                    "y_true_2": float(Y_val[row_idx, 1]),
                    "y_true_3": float(Y_val[row_idx, 2]),
                    "y_true_4": float(Y_val[row_idx, 3]),
                    "y_test5_before_1": float(y_test5_before[row_idx, 0]),
                    "y_test5_before_2": float(y_test5_before[row_idx, 1]),
                    "y_test5_before_3": float(y_test5_before[row_idx, 2]),
                    "y_test5_before_4": float(y_test5_before[row_idx, 3]),
                    "y_test5_refined_1": float(y_test5_refined[row_idx, 0]),
                    "y_test5_refined_2": float(y_test5_refined[row_idx, 1]),
                    "y_test5_refined_3": float(y_test5_refined[row_idx, 2]),
                    "y_test5_refined_4": float(y_test5_refined[row_idx, 3]),
                    "y_test5_snapped_1": float(y_test5_snapped[row_idx, 0]),
                    "y_test5_snapped_2": float(y_test5_snapped[row_idx, 1]),
                    "y_test5_snapped_3": float(y_test5_snapped[row_idx, 2]),
                    "y_test5_snapped_4": float(y_test5_snapped[row_idx, 3]),
                    "y_forest_1": float(y_forest[row_idx, 0]),
                    "y_forest_2": float(y_forest[row_idx, 1]),
                    "y_forest_3": float(y_forest[row_idx, 2]),
                    "y_forest_4": float(y_forest[row_idx, 3]),
                    "y_forest_refined_1": float(y_forest_refined[row_idx, 0]),
                    "y_forest_refined_2": float(y_forest_refined[row_idx, 1]),
                    "y_forest_refined_3": float(y_forest_refined[row_idx, 2]),
                    "y_forest_refined_4": float(y_forest_refined[row_idx, 3]),
                }
            )

    # Plots (test5-like + comparison)
    out_path = out_dir / "z_abs_error_hist.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_z_abs_err, bins=40, color="tab:green", alpha=0.8)
    ax.set_xlabel("Absolute error |z_test5 - z|")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of absolute z errors (test5 snapped)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "weighted_ssd_vs_abs_z_error.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        all_weighted_ssd,
        all_z_abs_err,
        s=18,
        alpha=0.7,
        c=all_z_var,
        cmap="viridis",
    )
    ax.set_xlabel("Best weighted SSD (grid search)")
    ax.set_ylabel("Absolute error |z_test5 - z|")
    ax.set_title("weighted_ssd vs absolute z error")
    fig.colorbar(sc, ax=ax, label="Var(z) for K lowest weighted_ssd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "weighted_ssd_vs_abs_z_error_by_refine_delta.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        all_weighted_ssd,
        all_z_abs_err,
        s=18,
        alpha=0.7,
        c=all_z_refine_delta,
        cmap="plasma",
    )
    ax.set_xlabel("Best weighted SSD (grid search)")
    ax.set_ylabel("Absolute error |z_test5 - z|")
    ax.set_title("weighted_ssd vs absolute z error (color: |z_refined - z_init|)")
    fig.colorbar(sc, ax=ax, label="|z_refined - z_init|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "abs_z_error_before_vs_after.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_abs_err_before, all_z_abs_err, s=18, alpha=0.7, color="tab:purple")
    lim = max(max(all_z_abs_err_before), max(all_z_abs_err))
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute error |z_init - z|")
    ax.set_ylabel("Absolute error |z_snapped - z|")
    ax.set_title("Absolute z error: before vs snapped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "abs_z_error_refined_vs_snapped.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_abs_err_refined, all_z_abs_err, s=18, alpha=0.7, color="tab:orange")
    lim = max(max(all_z_abs_err_refined), max(all_z_abs_err))
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute error |z_refined - z|")
    ax.set_ylabel("Absolute error |z_snapped - z|")
    ax.set_title("Absolute z error: refined vs snapped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "abs_z_error_vs_abs_y_error.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_abs_err, all_y_abs_err, s=18, alpha=0.7, color="tab:blue")
    ax.set_xlabel("Absolute error |z_test5 - z|")
    ax.set_ylabel("Absolute error |y_test5 - y|")
    ax.set_title("Absolute z error vs absolute y error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "weighted_ssd_vs_weighted_ssd_true.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        all_weighted_ssd,
        all_weighted_ssd_true,
        s=18,
        alpha=0.7,
        c=all_z_abs_err,
        cmap="viridis",
    )
    lim = max(max(all_weighted_ssd), max(all_weighted_ssd_true))
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Best weighted SSD (grid search)")
    ax.set_ylabel("Weighted SSD at z_true")
    ax.set_title("Weighted SSD vs weighted SSD (z_true)")
    fig.colorbar(sc, ax=ax, label="Absolute error |z_test5 - z|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "weighted_ssd_vs_abs_z_error_by_ssd_ratio.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        all_weighted_ssd,
        all_z_abs_err,
        s=18,
        alpha=0.7,
        c=all_ssd_ratio,
        cmap="viridis",
    )
    ax.set_xlabel("Best weighted SSD (grid search)")
    ax.set_ylabel("Absolute error |z_test5 - z|")
    ax.set_title("Weighted SSD vs absolute z error (color: unweighted/weighted)")
    fig.colorbar(sc, ax=ax, label="Unweighted SSD / Weighted SSD")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    if rows:
        df_out = pd.DataFrame(rows)
        out_path = out_dir / "predictions_partial_final.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")

    out_path = out_dir / "abs_z_error_test5_vs_test9_refined.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_abs_err, all_z_abs_err_forest_refined, s=18, alpha=0.7, color="tab:cyan")
    lim = max(max(all_z_abs_err), max(all_z_abs_err_forest_refined))
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute error |z_test5 - z|")
    ax.set_ylabel("Absolute error |z_forest_refined - z|")
    ax.set_title("Absolute z error: test5 vs test9 (refined)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    all_z_test5 = np.array(all_z_test5)
    all_z_forest_refined = np.array(all_z_forest_refined)
    all_y_val = np.vstack(all_y_val)

    diffs = np.abs(all_z_test5 - all_z_forest_refined)
    y_test5 = reconstruct_targets(all_z_test5)
    err_test5 = np.linalg.norm(y_test5 - all_y_val, axis=1)

    order = np.argsort(diffs)
    diffs_sorted = diffs[order]
    err_test5_sorted = err_test5[order]

    cum_test5 = np.cumsum(err_test5_sorted)
    total_test5 = float(cum_test5[-1])
    n = len(diffs_sorted)

    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for gamma in np.arange(0.1, 1.01, 0.1):
        z_mix = (1.0 - gamma) * all_z_test5 + gamma * all_z_forest_refined
        y_mix = reconstruct_targets(z_mix)
        err_mix = np.linalg.norm(y_mix - all_y_val, axis=1)
        err_mix_sorted = err_mix[order]
        cum_mix = np.cumsum(err_mix_sorted)

        mee_by_alpha = []
        for k in range(n):
            total_err = cum_mix[k] + (total_test5 - cum_test5[k])
            mee_by_alpha.append(total_err / n)

        out_path = sweep_dir / f"alpha_vs_mee_gamma_{gamma:.1f}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(diffs_sorted, mee_by_alpha, color="tab:blue", linewidth=2)
        ax.set_xlabel("alpha")
        ax.set_ylabel("MEE (avg if |z_test5 - z_forest| < alpha)")
        ax.set_title(f"alpha vs MEE (gamma={gamma:.1f})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time() - start:.2f}s")
