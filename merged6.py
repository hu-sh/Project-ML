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
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]
SOFTMAX_K = 256
SIGMA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
K_SWEEP_MAX = 256
K_SWEEP_SIGMA = 0.05


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


def _refine_with_lambda_sweep(z0, pc2, Y_ref, T, const, terms, z_min, z_max):
    best = None
    for lam in LAMBDA_SWEEP:
        z_ref = np.array(
            [_refine_z_reg(z_i, pc2_i, T, const, terms, z_min, z_max, lam) for z_i, pc2_i in zip(z0, pc2)]
        )
        y_pred = np.array([_reconstruct_y_from_z(z_i) for z_i in z_ref])
        mee = np.mean(np.linalg.norm(y_pred - Y_ref, axis=1))
        if best is None or mee < best[0]:
            best = (mee, lam, z_ref)
    return best


def _refine_with_lambda(z0, pc2, T, const, terms, z_min, z_max, lam):
    return np.array(
        [_refine_z_reg(z_i, pc2_i, T, const, terms, z_min, z_max, lam) for z_i, pc2_i in zip(z0, pc2)]
    )


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
    all_z_errors = []
    all_z_vals = []
    errors_by_sigma = None
    errors_by_sigma_no_ref = None
    all_errors_refined = []
    errors_by_k_sweep = None

    out_dir = Path("plots/6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_random_forest_abs_error_hist.png"

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
        pc2_train = X_train_pca[:, 1]
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
        if errors_by_sigma is None:
            errors_by_sigma = {sigma: [] for sigma in SIGMA_GRID}
        if errors_by_sigma_no_ref is None:
            errors_by_sigma_no_ref = {sigma: [] for sigma in SIGMA_GRID}
        if errors_by_k_sweep is None:
            errors_by_k_sweep = {k: [] for k in range(1, K_SWEEP_MAX + 1)}

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

        z_random_forest_train = []
        for i in range(len(X_train)):
            z_rf, _ = predict_z_with_pca(X_train[i])
            z_random_forest_train.append(z_rf)
        z_random_forest_train = np.array(z_random_forest_train)
        z_min_ref = float(z_train.min())
        z_max_ref = float(z_train.max())
        _, best_lam, _ = _refine_with_lambda_sweep(
            z_random_forest_train, pc2_train, Y_train, T, const, terms, z_min_ref, z_max_ref
        )

        z_candidates_by_sigma = {sigma: [] for sigma in SIGMA_GRID}
        z_candidates_by_k_sweep = {k: [] for k in range(1, K_SWEEP_MAX + 1)}
        diff_best_val = []
        for i in range(len(X_val)):
            x_scaled = scaler_X.transform([X_val[i]])
            x_pca = pca.transform(x_scaled)[0]
            dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
            best_idx = int(np.argmin(dists))
            diff_best_val.append(X_grid_pca_pred[best_idx] - x_pca)
            k = min(SOFTMAX_K, len(dists))
            idx_k = np.argpartition(dists, k - 1)[:k]
            d_sel = dists[idx_k]
            z_sel = z_grid[idx_k]
            for sigma in SIGMA_GRID:
                x = -(d_sel ** 2) / (2 * sigma ** 2)
                x -= x.max()
                w = np.exp(x)
                w /= w.sum()
                z_candidates_by_sigma[sigma].append(float(np.sum(w * z_sel)))

            k_sweep = min(K_SWEEP_MAX, len(dists))
            idx_k_sweep = np.argpartition(dists, k_sweep - 1)[:k_sweep]
            d_k_sweep = dists[idx_k_sweep]
            z_k_sweep = z_grid[idx_k_sweep]
            order = np.argsort(d_k_sweep)
            d_k_sweep = d_k_sweep[order]
            z_k_sweep = z_k_sweep[order]
            for k in range(1, k_sweep + 1):
                d_k = d_k_sweep[:k]
                z_k = z_k_sweep[:k]
                x = -(d_k ** 2) / (2 * K_SWEEP_SIGMA ** 2)
                x -= x.max()
                w = np.exp(x)
                w /= w.sum()
                z_candidates_by_k_sweep[k].append(float(np.sum(w * z_k)))

        z_random_forest_val = []
        for i in range(len(X_val)):
            z_rf, _ = predict_z_with_pca(X_val[i])
            z_random_forest_val.append(z_rf)
        z_random_forest_val = np.array(z_random_forest_val)

        errors_nn = []
        for i in range(len(X_val)):
            y_est = _reconstruct_y_from_z(z_random_forest_val[i])
            err = np.linalg.norm(y_est - Y_val[i])
            errors_nn.append(err)
        mee_nn = np.mean(errors_nn)
        print(f"MEE Fold {fold_idx} (closest, no refine): {mee_nn:.5f}")
        all_errors_nn.extend(errors_nn)

        z_random_forest_val_refined = _refine_with_lambda(
            z_random_forest_val, pc2_val, T, const, terms, z_min_ref, z_max_ref, best_lam
        )
        z_errors = np.abs(z_random_forest_val_refined - z_val)
        for i in range(len(X_val)):
            y_est = _reconstruct_y_from_z(z_random_forest_val_refined[i])
            err = np.linalg.norm(y_est - Y_val[i])
            errors.append(err)
        mee_refined = np.mean(errors)
        print(f"MEE Fold {fold_idx} (closest, refined): {mee_refined:.5f}")
        all_errors_refined.extend(errors)

        mee_by_sigma = {}
        mee_by_sigma_no_ref = {}
        for sigma in SIGMA_GRID:
            z_candidates = np.array(z_candidates_by_sigma[sigma])
            z_refined = _refine_with_lambda(
                z_candidates, pc2_val, T, const, terms, z_min_ref, z_max_ref, best_lam
            )
            y_est = np.array([_reconstruct_y_from_z(z_i) for z_i in z_refined])
            err_sigma = np.linalg.norm(y_est - Y_val, axis=1)
            errors_by_sigma[sigma].extend(err_sigma)
            mee_by_sigma[sigma] = np.mean(err_sigma)

            y_est_no_ref = np.array([_reconstruct_y_from_z(z_i) for z_i in z_candidates])
            err_sigma_no_ref = np.linalg.norm(y_est_no_ref - Y_val, axis=1)
            errors_by_sigma_no_ref[sigma].extend(err_sigma_no_ref)
            mee_by_sigma_no_ref[sigma] = np.mean(err_sigma_no_ref)
        print(
            "Softmax MEE by sigma (fold "
            + str(fold_idx)
            + "): "
            + ", ".join([f"s={s}: {mee_by_sigma[s]:.5f}" for s in SIGMA_GRID])
        )
        print(
            "Softmax MEE by sigma (no refine, fold "
            + str(fold_idx)
            + "): "
            + ", ".join([f"s={s}: {mee_by_sigma_no_ref[s]:.5f}" for s in SIGMA_GRID])
        )

        for k in range(1, K_SWEEP_MAX + 1):
            z_candidates = np.array(z_candidates_by_k_sweep[k])
            z_refined = _refine_with_lambda(
                z_candidates, pc2_val, T, const, terms, z_min_ref, z_max_ref, best_lam
            )
            y_est = np.array([_reconstruct_y_from_z(z_i) for z_i in z_refined])
            err_k = np.linalg.norm(y_est - Y_val, axis=1)
            errors_by_k_sweep[k].extend(err_k)

        mee = np.mean(errors)
        print(f"MEE Fold {fold_idx}: {mee:.5f}")
        all_errors.extend(errors)
        all_z_errors.extend(z_errors)
        all_z_vals.extend(z_val)

        diff_best_val = np.asarray(diff_best_val)
        out_path = out_dir / f"pca_diff_comp1_vs_comp2_fold_{fold_idx}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(diff_best_val[:, 0], diff_best_val[:, 1], s=18, alpha=0.7, color="tab:purple")
        ax.set_xlabel("Component 1 of (X_grid_pca_pred - x_pca)")
        ax.set_ylabel("Component 2 of (X_grid_pca_pred - x_pca)")
        ax.set_title(f"Diff components (fold {fold_idx})")
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
    if errors_by_sigma is not None:
        mee_by_sigma_all = {s: float(np.mean(errors_by_sigma[s])) for s in SIGMA_GRID}
        print(
            "Softmax MEE by sigma (overall): "
            + ", ".join([f"s={s}: {mee_by_sigma_all[s]:.5f}" for s in SIGMA_GRID])
        )
    if errors_by_sigma_no_ref is not None:
        mee_by_sigma_no_ref_all = {s: float(np.mean(errors_by_sigma_no_ref[s])) for s in SIGMA_GRID}
        print(
            "Softmax MEE by sigma (no refine, overall): "
            + ", ".join([f"s={s}: {mee_by_sigma_no_ref_all[s]:.5f}" for s in SIGMA_GRID])
        )
    if errors_by_k_sweep is not None:
        mee_by_k = np.array([np.mean(errors_by_k_sweep[k]) for k in range(1, K_SWEEP_MAX + 1)])
        out_path = out_dir / "mee_by_k_softmax_sigma_0p05.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(1, K_SWEEP_MAX + 1), mee_by_k, color="tab:blue", linewidth=2)
        ax.set_xlabel("Number of nearest points (K)")
        ax.set_ylabel("MEE")
        ax.set_title("MEE vs K (softmax, sigma=0.05)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}", flush=True)

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
