import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import differential_evolution
from scipy import stats


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"

N_COMPONENTS = 6
N_PCA_COMPONENTS = 5
K_NEIGHBORS = 2
RIDGE_ALPHA = 0.0000001

LOCAL_PCA_COMPONENTS = 4
LOCAL_RIDGE_ALPHA = 0.01


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


def _hist_plot(values, out_path, xlabel, title, color="tab:blue", bins=40):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, color=color, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}", flush=True)


def main():
    df = pd.read_csv(FILE_PATH, comment="#", header=None, index_col=0)
    X = df.iloc[:, 0:12].values
    Y = df.iloc[:, 12:16].values
    z = Y[:, 2] - Y[:, 3]

    rng_seed = int(time.time_ns() % 1_000_000)
    print(f"random_state: {rng_seed}", flush=True)

    err_z_pred = []
    err_z_pred_pca = []
    err_z_est_raw = []
    err_y_final = []

    z_pred_all = []
    z_pred_pca_all = []
    z_est_raw_all = []
    z_val_all = []
    y_final_pred_all = []
    y_val_all = []
    fold_ids_all = []

    fit_params_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=rng_seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        z_train, z_val = z[train_idx], z[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        max_components = max(N_COMPONENTS, N_PCA_COMPONENTS, 2)
        pca = PCA(n_components=max_components)
        X_train_pca_full = pca.fit_transform(X_train_scaled)
        X_val_pca_full = pca.transform(X_val_scaled)

        X_train_pca = X_train_pca_full[:, :N_COMPONENTS]

        # y_final_pred (no refinement)
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
        print(f"Fold {fold_idx} MEE differential evolution: {mee_de:.5f}", flush=True)
        y_final_pred_train = _local_ridge_predict(
            best_weights, X_train_scaled, X_train_scaled, X_train_pca_local, X_train_pca_local, Y_train
        )

        # z_est_raw (PCA + RF)
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

        z_est_raw = np.array([predict_z_with_pca(x) for x in X_val])
        z_est_raw_train = np.array([predict_z_with_pca(x) for x in X_train])

        # z_pred (KNN + local ridge)
        X_train_pca_knn = X_train_pca_full[:, :N_PCA_COMPONENTS]
        X_val_pca_knn = X_val_pca_full[:, :N_PCA_COMPONENTS]

        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="ball_tree").fit(X_train_pca_knn)
        _, indices = nbrs.kneighbors(X_val_pca_knn)

        z_pred = []
        for i in range(len(X_val)):
            idx = indices[i]
            X_loc = X_train_scaled[idx]
            z_loc = z_train[idx]
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_loc, z_loc)
            pred = model.predict([X_val_scaled[i]])[0]
            z_pred.append(pred)
        z_pred = np.array(z_pred).flatten()
        _, indices_train = nbrs.kneighbors(X_train_pca_knn)
        z_pred_train = []
        for i in range(len(X_train)):
            idx = indices_train[i]
            X_loc = X_train_scaled[idx]
            z_loc = z_train[idx]
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_loc, z_loc)
            pred = model.predict([X_train_scaled[i]])[0]
            z_pred_train.append(pred)
        z_pred_train = np.array(z_pred_train).flatten()

        # z_pred_pca (linear model on PC1/PC2)
        pc_val = X_val_pca_full[:, :2]
        pc_intercept = -7.4426
        pc_betas = np.array([-10.9003, 2.6545])
        z_pred_pca = pc_intercept + pc_val @ pc_betas
        pc_train = X_train_pca_full[:, :2]
        z_pred_pca_train = pc_intercept + pc_train @ pc_betas

        err_z_pred.append(z_pred - z_val)
        err_z_pred_pca.append(z_pred_pca - z_val)
        err_z_est_raw.append(z_est_raw - z_val)
        err_y_final.append(np.linalg.norm(y_final_pred - Y_val, axis=1))

        z_pred_all.append(z_pred)
        z_pred_pca_all.append(z_pred_pca)
        z_est_raw_all.append(z_est_raw)
        z_val_all.append(z_val)
        y_final_pred_all.append(y_final_pred)
        y_val_all.append(Y_val)
        fold_ids_all.append(np.full_like(z_val, fold_idx, dtype=int))

        z_pred_err_train = z_pred_train - z_train
        z_pred_pca_err_train = z_pred_pca_train - z_train
        z_est_raw_err_train = z_est_raw_train - z_train
        y_final_err_train = np.linalg.norm(y_final_pred_train - Y_train, axis=1)

        gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(y_final_err_train, floc=0.0)
        t_est_raw_df, t_est_raw_loc, t_est_raw_scale = stats.t.fit(z_est_raw_err_train)
        norm_mu, norm_sigma = stats.norm.fit(z_pred_pca_err_train)
        t_pred_df, t_pred_loc, t_pred_scale = stats.t.fit(z_pred_err_train)
        fit_params_rows.append(
            {
                "fold": fold_idx,
                "gamma_k": gamma_shape,
                "gamma_loc": gamma_loc,
                "gamma_scale": gamma_scale,
                "t_est_raw_df": t_est_raw_df,
                "t_est_raw_loc": t_est_raw_loc,
                "t_est_raw_scale": t_est_raw_scale,
                "norm_mu": norm_mu,
                "norm_sigma": norm_sigma,
                "t_pred_df": t_pred_df,
                "t_pred_loc": t_pred_loc,
                "t_pred_scale": t_pred_scale,
            }
        )

    out_dir = Path("plots/errors")
    out_dir.mkdir(parents=True, exist_ok=True)

    z_pred_all = np.concatenate(z_pred_all)
    z_pred_pca_all = np.concatenate(z_pred_pca_all)
    z_est_raw_all = np.concatenate(z_est_raw_all)
    z_val_all = np.concatenate(z_val_all)
    y_final_pred_all = np.vstack(y_final_pred_all)
    y_val_all = np.vstack(y_val_all)

    errors_df = pd.DataFrame(
        {
            "fold": np.concatenate(fold_ids_all),
            "z_val": z_val_all,
            "z_pred": z_pred_all,
            "z_pred_pca": z_pred_pca_all,
            "z_est_raw": z_est_raw_all,
            "z_pred_err": z_pred_all - z_val_all,
            "z_pred_pca_err": z_pred_pca_all - z_val_all,
            "z_est_raw_err": z_est_raw_all - z_val_all,
            "y_err_norm": np.linalg.norm(y_final_pred_all - y_val_all, axis=1),
            "y_val_1": y_val_all[:, 0],
            "y_val_2": y_val_all[:, 1],
            "y_val_3": y_val_all[:, 2],
            "y_val_4": y_val_all[:, 3],
            "y_pred_1": y_final_pred_all[:, 0],
            "y_pred_2": y_final_pred_all[:, 1],
            "y_pred_3": y_final_pred_all[:, 2],
            "y_pred_4": y_final_pred_all[:, 3],
        }
    )
    out_csv = out_dir / "predictions_cv.csv"
    errors_df.to_csv(out_csv, index=False)
    print(f"Saved predictions: {out_csv}", flush=True)

    fit_params_df = pd.DataFrame(fit_params_rows)
    fit_params_csv = out_dir / "fit_params_by_fold.csv"
    fit_params_df.to_csv(fit_params_csv, index=False)
    print(f"Saved fold fit params: {fit_params_csv}", flush=True)

    z_pred_err_all = np.concatenate(err_z_pred)
    z_pred_pca_err_all = np.concatenate(err_z_pred_pca)
    z_est_raw_err_all = np.concatenate(err_z_est_raw)
    y_final_err_all = np.concatenate(err_y_final)

    _hist_plot(
        z_pred_err_all,
        out_dir / "z_pred_signed_error.png",
        "Signed error (z_pred - z)",
        "Histogram of z_pred signed errors",
        color="tab:green",
    )
    _hist_plot(
        z_pred_pca_err_all,
        out_dir / "z_pred_pca_signed_error.png",
        "Signed error (z_pred_pca - z)",
        "Histogram of z_pred_pca signed errors",
        color="tab:purple",
    )
    _hist_plot(
        z_est_raw_err_all,
        out_dir / "z_est_raw_signed_error.png",
        "Signed error (z_est_raw - z)",
        "Histogram of z_est_raw signed errors",
        color="tab:blue",
    )
    _hist_plot(
        y_final_err_all,
        out_dir / "y_final_pred_norm_error.png",
        "|y_final_pred - y| (norm)",
        "Histogram of |y_final_pred - y|",
        color="tab:orange",
    )

    gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(y_final_err_all, floc=0.0)
    t_est_raw_df, t_est_raw_loc, t_est_raw_scale = stats.t.fit(z_est_raw_err_all)
    norm_mu, norm_sigma = stats.norm.fit(z_pred_pca_err_all)
    t_pred_df, t_pred_loc, t_pred_scale = stats.t.fit(z_pred_err_all)

    fit_path = out_dir / "fit_params.txt"
    with open(fit_path, "w", encoding="utf-8") as f:
        f.write("y_final_pred_norm ~ Gamma(k, loc, scale)\n")
        f.write(f"k={gamma_shape:.6f}, loc={gamma_loc:.6f}, scale={gamma_scale:.6f}\n\n")
        f.write("z_est_raw_err ~ StudentT(df, loc, scale)\n")
        f.write(f"df={t_est_raw_df:.6f}, loc={t_est_raw_loc:.6f}, scale={t_est_raw_scale:.6f}\n\n")
        f.write("z_pred_pca_err ~ Normal(mu, sigma)\n")
        f.write(f"mu={norm_mu:.6f}, sigma={norm_sigma:.6f}\n\n")
        f.write("z_pred_err ~ StudentT(df, loc, scale)\n")
        f.write(f"df={t_pred_df:.6f}, loc={t_pred_loc:.6f}, scale={t_pred_scale:.6f}\n")
    print(f"Saved fit params: {fit_path}", flush=True)


if __name__ == "__main__":
    main()
