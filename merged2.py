import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import merged


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
TRAIN_SPLIT = 0.8  # train/val = 45/55
VAL_TUNE_SPLIT = 0.8  # tuning/eval within val

LAMBDA_SWEEP = merged.LAMBDA_SWEEP
PC_INTERCEPT = merged.PC_INTERCEPT
PC_BETAS = merged.PC_BETAS

W_SWEEP = np.linspace(0.0, 10.0, 21)
BIAS_SWEEP = np.linspace(-5.0, 5.0, 21)


def _mee(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main() -> dict:
    random_state = int(time.time_ns() % 1_000_000)
    print(f"random_state: {random_state}", flush=True)

    raw = pd.read_csv(FILE_PATH, comment="#", header=None, index_col=0)
    X = raw.iloc[:, 0:12].values
    Y = raw.iloc[:, 12:16].values
    z = Y[:, 2] - Y[:, 3]

    X_train, X_val, Y_train, Y_val, z_train, z_val = train_test_split(
        X, Y, z, test_size=(1 - TRAIN_SPLIT), random_state=random_state
    )

    # Shared scaler + PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    max_components = max(merged.N_COMPONENTS, merged.N_PCA_COMPONENTS, 2)
    pca = PCA(n_components=max_components)
    X_train_pca_full = pca.fit_transform(X_train_scaled)
    X_val_pca_full = pca.transform(X_val_scaled)

    # y_final_pred via local ridge optimization (test3.py logic)
    X_train_pca_local = X_train_pca_full[:, :merged.LOCAL_PCA_COMPONENTS]
    X_val_pca_local = X_val_pca_full[:, :merged.LOCAL_PCA_COMPONENTS]
    bounds = [(0, 10)] * merged.LOCAL_PCA_COMPONENTS
    result = differential_evolution(
        merged._objective_function,
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
    weights = result.x
    y_final_pred = merged._local_ridge_predict(
        weights, X_train_scaled, X_val_scaled, X_train_pca_local, X_val_pca_local, Y_train
    )
    mee_de = _mee(Y_val, y_final_pred)
    print(f"MEE y_final_pred: {mee_de:.5f}", flush=True)

    # z_est (test9-style)
    X_train_pca = X_train_pca_full[:, :merged.N_COMPONENTS]
    X_val_pca = X_val_pca_full[:, :merged.N_COMPONENTS]
    model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
    model_rf.fit(z_train.reshape(-1, 1), X_train_pca)
    z_min_grid, z_max_grid = z_train.min() - 10, z_train.max() + 10
    z_grid = np.linspace(z_min_grid, z_max_grid, 500000)
    X_grid_pca_pred = model_rf.predict(z_grid.reshape(-1, 1))
    z_est = []
    for x in X_val:
        x_scaled = scaler.transform([x])
        x_pca = pca.transform(x_scaled)[:, :merged.N_COMPONENTS]
        dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
        z_est.append(z_grid[int(np.argmin(dists))])
    z_est = np.array(z_est)
    mee_z_est_raw = _mee(Y_val, merged._reconstruct_y(z_est))
    print(f"MEE z_est (raw): {mee_z_est_raw:.5f}", flush=True)

    # z_pred (KNN local ridge on z)
    X_train_pca_knn = X_train_pca_full[:, :merged.N_PCA_COMPONENTS]
    X_val_pca_knn = X_val_pca_full[:, :merged.N_PCA_COMPONENTS]
    nbrs = NearestNeighbors(n_neighbors=merged.K_NEIGHBORS, algorithm="ball_tree").fit(X_train_pca_knn)
    _, indices = nbrs.kneighbors(X_val_pca_knn)
    z_pred = []
    for i in range(len(X_val)):
        idx = indices[i]
        X_loc = X_train_scaled[idx]
        z_loc = z_train[idx]
        model = Ridge(alpha=merged.RIDGE_ALPHA)
        model.fit(X_loc, z_loc)
        pred = model.predict([X_val_scaled[i]])[0]
        z_pred.append(pred)
    z_pred = np.array(z_pred).flatten()
    mee_z_pred_raw = _mee(Y_val, merged._reconstruct_y(z_pred))
    print(f"MEE z_pred (raw): {mee_z_pred_raw:.5f}", flush=True)

    # z_pred_pc
    pc_val = X_val_pca_full[:, :2]
    pc2_val = pc_val[:, 1]
    z_pred_pc = PC_INTERCEPT + pc_val @ PC_BETAS

    # Split validation for tuning
    val_idx = np.arange(len(Y_val))
    tune_idx, eval_idx = train_test_split(val_idx, test_size=(1 - VAL_TUNE_SPLIT), random_state=42)

    T, const, terms = merged._parse_fit(merged.FIT_PATH)
    z_min = float(z.min())
    z_max = float(z.max())

    def refine_z(z_input, idx, lam):
        return np.array(
            [
                merged._refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                for z0, pc2 in zip(z_input[idx], pc2_val[idx])
            ]
        )

    # Tune lambda for z_est
    best = None
    for lam in LAMBDA_SWEEP:
        z_ref = refine_z(z_est, tune_idx, lam)
        mee = _mee(Y_val[tune_idx], merged._reconstruct_y(z_ref))
        if best is None or mee < best[0]:
            best = (mee, lam)
    _, lam_est = best
    z_est_ref = refine_z(z_est, val_idx, lam_est)
    mee_z_est_ref = _mee(Y_val, merged._reconstruct_y(z_est_ref))
    print(f"MEE z_est (refined): {mee_z_est_ref:.5f}", flush=True)

    # Tune lambda for z_pred
    best = None
    for lam in LAMBDA_SWEEP:
        z_ref = refine_z(z_pred, tune_idx, lam)
        mee = _mee(Y_val[tune_idx], merged._reconstruct_y(z_ref))
        if best is None or mee < best[0]:
            best = (mee, lam)
    _, lam_pred = best
    z_pred_ref = refine_z(z_pred, val_idx, lam_pred)
    mee_z_pred_ref = _mee(Y_val, merged._reconstruct_y(z_pred_ref))
    print(f"MEE z_pred (refined): {mee_z_pred_ref:.5f}", flush=True)

    # Score-based selection (tune weight and bias on tune split)
    d_est = np.array([merged._d_perp(merged._y_of_z(np.array([z]))[0]) for z in z_est_ref])
    d_pred = np.array([merged._d_perp(merged._y_of_z(np.array([z]))[0]) for z in z_pred_ref])
    dist_est = (z_pred_pc - z_est_ref) ** 2
    dist_pred = (z_pred_pc - z_pred_ref) ** 2

    best = None
    for w in W_SWEEP:
        for bias in BIAS_SWEEP:
            score_est = -w * (d_est ** 2) - dist_est + bias
            score_pred = -w * (d_pred ** 2) - dist_pred
            pick_est = score_est >= score_pred
            z_sel = np.where(pick_est, z_est_ref, z_pred_ref)
            mee = _mee(Y_val[tune_idx], merged._reconstruct_y(z_sel[tune_idx]))
            if best is None or mee < best[0]:
                best = (mee, w, bias)

    _, w_opt, bias_opt = best
    score_est = -w_opt * (d_est ** 2) - dist_est + bias_opt
    score_pred = -w_opt * (d_pred ** 2) - dist_pred
    pick_est = score_est >= score_pred
    z_sel = np.where(pick_est, z_est_ref, z_pred_ref)
    mee_sel = _mee(Y_val[eval_idx], merged._reconstruct_y(z_sel[eval_idx]))
    print(f"Selected MEE (eval): {mee_sel:.5f}", flush=True)

    return {
        "mee_de": mee_de,
        "mee_z_est": mee_z_est_ref,
        "mee_z_ref": mee_z_pred_ref,
        "mee_ens_refined": mee_z_pred_ref,
        "mee_selected": mee_sel,
    }


if __name__ == "__main__":
    main()
