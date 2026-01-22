import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
FIT_PATH = "fit.txt"
W_THEORETICAL = (1 + 1 / np.sqrt(2)) / 3
LAMBDA_SWEEP = [1.0]
GRID_STEPS = 201
GRID_SIZE = 5000
K_SSD = 100
N_COMPONENTS = 6
GAMMA = 0.1
ALPHA_THRESHOLD = 1.5
GAMMA_GRID = np.linspace(0.0, 0.5, 51)


def calculate_mee(y_true, y_pred):
    errors = y_true - y_pred
    dist = np.sqrt(np.sum(errors ** 2, axis=1))
    return np.mean(dist)


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


def weighted_ssd_obj(z, x_obs, coefs_mat, intercepts, weights):
    z_val = float(np.atleast_1d(z)[0])
    feats = get_physical_features(np.array([z_val]))
    x_pred = feats @ coefs_mat.T + intercepts
    diff_sq = (x_pred[0] - x_obs) ** 2
    return float(np.sum(diff_sq * weights))


def train_harmonic_forward(X_train, z_train):
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
    return weights, coefs_mat, intercepts, z_grid, grid_preds, z_min, z_max


def recover_z_harmonic(X_eval, z_grid, grid_preds, coefs_mat, intercepts, weights, z_min, z_max):
    z_init = []
    for i in range(len(X_eval)):
        x_obs = X_eval[i]
        diff_sq = (grid_preds - x_obs) ** 2
        weighted_ssd_vals = np.sum(diff_sq * weights, axis=1)
        z_init.append(z_grid[int(np.argmin(weighted_ssd_vals))])
    z_init = np.array(z_init)

    z_rec = []
    for i in range(len(X_eval)):
        x_obs = X_eval[i]
        result = minimize(
            weighted_ssd_obj,
            x0=np.array([float(z_init[i])]),
            args=(x_obs, coefs_mat, intercepts, weights),
            method="L-BFGS-B",
            bounds=[(z_min, z_max)],
        )
        if result.success:
            z_rec.append(float(result.x[0]))
        else:
            z_rec.append(float(z_init[i]))
    return np.array(z_rec), z_init


def fit_pc2_model(X_train):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_sc)
    return scaler, pca, X_train_pca[:, 1]


def transform_pc2(X_eval, scaler, pca):
    X_eval_sc = scaler.transform(X_eval)
    X_eval_pca = pca.transform(X_eval_sc)
    return X_eval_pca[:, 1]


def train_rf_pca_inversion(X_train, z_train):
    scaler_rf = StandardScaler()
    X_train_scaled = scaler_rf.fit_transform(X_train)
    pca_rf = PCA(n_components=N_COMPONENTS)
    X_train_pca_rf = pca_rf.fit_transform(X_train_scaled)
    model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
    model_rf.fit(z_train.reshape(-1, 1), X_train_pca_rf)
    z_min_rf, z_max_rf = z_train.min() - 10, z_train.max() + 10
    z_grid_rf = np.linspace(z_min_rf, z_max_rf, 500000)
    X_grid_pca_pred = model_rf.predict(z_grid_rf.reshape(-1, 1))
    return scaler_rf, pca_rf, z_grid_rf, X_grid_pca_pred, z_min_rf, z_max_rf


def recover_z_rf(X_eval, scaler_rf, pca_rf, z_grid_rf, X_grid_pca_pred):
    z_out = []
    for i in range(len(X_eval)):
        x_scaled = scaler_rf.transform([X_eval[i]])
        x_pca = pca_rf.transform(x_scaled)[0]
        dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
        z_out.append(z_grid_rf[int(np.argmin(dists))])
    return np.array(z_out)


def select_best_lambda(z_init, pc2_train, Y_train, T, const, terms, z_min, z_max):
    best = None
    for lam in LAMBDA_SWEEP:
        z_ref = np.array(
            [
                refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                for z0, pc2 in zip(z_init, pc2_train)
            ]
        )
        mee_train = calculate_mee(Y_train, reconstruct_targets(z_ref))
        if best is None or mee_train < best[0]:
            best = (mee_train, lam)
    return best[1]


def run_cv():
    df = pd.read_csv(FILE_PATH, comment="#", header=None)
    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values
    z_all = Y[:, 2] - Y[:, 3]

    T, const, terms = parse_fit(FIT_PATH)

    random_state = 56
    print(f"random_state: {random_state}")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    all_z_recovered = []
    all_z_recovered_refined = []
    all_z_forest = []
    all_z_forest_refined = []
    all_z_ens = []
    all_z_ens_refined = []
    all_y_val = []

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

        z_recovered_grid = []
        for i in range(len(X_val)):
            x_obs = X_val[i]
            diff_sq = (grid_preds - x_obs) ** 2
            weighted_ssd_vals = np.sum(diff_sq * weights, axis=1)
            best_grid_idx = int(np.argmin(weighted_ssd_vals))
            z_recovered_grid.append(z_grid[best_grid_idx])
        z_recovered_grid = np.array(z_recovered_grid)

        z_recovered = []
        for i in range(len(X_val)):
            x_obs = X_val[i]
            z_init = float(z_recovered_grid[i])
            result = minimize(
                weighted_ssd_obj,
                x0=np.array([z_init]),
                args=(x_obs, coefs_mat, intercepts, weights),
                method="L-BFGS-B",
                bounds=[(z_min, z_max)],
            )
            if result.success:
                z_recovered.append(float(result.x[0]))
            else:
                z_recovered.append(z_init)
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

        # --- ensemble (conditional weighted average) ---
        diffs = np.abs(z_recovered_refined - z_forest_refined)
        use_weighted = diffs < ALPHA_THRESHOLD
        z_ens = np.where(
            use_weighted,
            (1.0 - GAMMA) * z_recovered_refined + GAMMA * z_forest_refined,
            z_recovered_refined,
        )
        z_ens_refined = np.array(
            [
                refine_z_reg(z0, pc2, T, const, terms, z_min_ref, z_max_ref, best_lam)
                for z0, pc2 in zip(z_ens, pc2_val)
            ]
        )

        mee_recovered = calculate_mee(Y_val, reconstruct_targets(z_recovered))
        mee_refined = calculate_mee(Y_val, reconstruct_targets(z_recovered_refined))
        mee_forest = calculate_mee(Y_val, reconstruct_targets(z_forest))
        mee_forest_refined = calculate_mee(Y_val, reconstruct_targets(z_forest_refined))
        mee_ens = calculate_mee(Y_val, reconstruct_targets(z_ens))
        mee_ens_refined = calculate_mee(Y_val, reconstruct_targets(z_ens_refined))

        print(f"  MEE z_recovered:         {mee_recovered:.6f}")
        print(f"  MEE z_recovered_refined: {mee_refined:.6f}")
        print(f"  MEE z_forest:            {mee_forest:.6f}")
        print(f"  MEE z_forest_refined:    {mee_forest_refined:.6f}")
        print(f"  MEE z_ens:               {mee_ens:.6f}")
        print(f"  MEE z_ens_refined:       {mee_ens_refined:.6f}")

        all_z_recovered.append(z_recovered)
        all_z_recovered_refined.append(z_recovered_refined)
        all_z_forest.append(z_forest)
        all_z_ens.append(z_ens)
        all_z_ens_refined.append(z_ens_refined)
        all_z_forest_refined.append(z_forest_refined)
        all_y_val.append(Y_val)

    all_z_recovered = np.concatenate(all_z_recovered)
    all_z_recovered_refined = np.concatenate(all_z_recovered_refined)
    all_z_forest = np.concatenate(all_z_forest)
    all_z_forest_refined = np.concatenate(all_z_forest_refined)
    all_z_ens = np.concatenate(all_z_ens)
    all_z_ens_refined = np.concatenate(all_z_ens_refined)
    all_y_val = np.vstack(all_y_val)

    total_mee_recovered = calculate_mee(all_y_val, reconstruct_targets(all_z_recovered))
    total_mee_refined = calculate_mee(all_y_val, reconstruct_targets(all_z_recovered_refined))
    total_mee_forest = calculate_mee(all_y_val, reconstruct_targets(all_z_forest))
    total_mee_forest_refined = calculate_mee(all_y_val, reconstruct_targets(all_z_forest_refined))
    total_mee_ens = calculate_mee(all_y_val, reconstruct_targets(all_z_ens))
    total_mee_ens_refined = calculate_mee(all_y_val, reconstruct_targets(all_z_ens_refined))

    print("\nTotal:")
    print(f"  MEE z_recovered:         {total_mee_recovered:.6f}")
    print(f"  MEE z_recovered_refined: {total_mee_refined:.6f}")
    print(f"  MEE z_forest:            {total_mee_forest:.6f}")
    print(f"  MEE z_forest_refined:    {total_mee_forest_refined:.6f}")
    print(f"  MEE z_ens:               {total_mee_ens:.6f}")
    print(f"  MEE z_ens_refined:       {total_mee_ens_refined:.6f}")

    diffs_all = np.abs(all_z_recovered_refined - all_z_forest_refined)
    y_pred_rec = reconstruct_targets(all_z_recovered_refined)
    err_rec = np.sqrt(np.sum((all_y_val - y_pred_rec) ** 2, axis=1))
    z_avg_fixed = (1.0 - GAMMA) * all_z_recovered_refined + GAMMA * all_z_forest_refined
    y_pred_avg_fixed = reconstruct_targets(z_avg_fixed)
    err_avg_fixed = np.sqrt(np.sum((all_y_val - y_pred_avg_fixed) ** 2, axis=1))

    order = np.argsort(diffs_all)
    diffs_sorted = diffs_all[order]
    delta_sorted = (err_avg_fixed - err_rec)[order]
    delta_cumsum = np.cumsum(delta_sorted)
    base_sum = float(np.sum(err_rec))

    alphas = diffs_sorted
    mee_alpha = (base_sum + delta_cumsum) / len(err_rec)

    mee_gamma = []
    for gamma in GAMMA_GRID:
        use_weighted = diffs_all < ALPHA_THRESHOLD
        z_ens_gamma = np.where(
            use_weighted,
            (1.0 - gamma) * all_z_recovered_refined + gamma * all_z_forest_refined,
            all_z_recovered_refined,
        )
        y_pred_gamma = reconstruct_targets(z_ens_gamma)
        err_gamma = np.sqrt(np.sum((all_y_val - y_pred_gamma) ** 2, axis=1))
        mee_gamma.append(float(np.mean(err_gamma)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(alphas, mee_alpha)
    axes[0].set_title("MEE vs alpha (all thresholds)")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("MEE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(GAMMA_GRID, mee_gamma, marker="o", color="tab:orange")
    axes[1].set_title("MEE vs gamma")
    axes[1].set_xlabel("gamma")
    axes[1].set_ylabel("MEE")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("plots/mee_alpha_gamma.png", dpi=150)


def main():
    run_cv()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time() - start:.2f}s")
