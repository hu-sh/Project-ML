import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re

# --- CONFIGURAZIONE ---
FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
# La frequenza fondamentale identificata
W_THEORETICAL = (1 + 1/np.sqrt(2)) / 3 
FIT_PATH = "fit.txt"
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]
GRID_STEPS = 201
K_SSD = 10

# Funzione per generare le features fisiche (Base Armonica)
def get_physical_features(z_arr):
    """
    Costruisce la matrice delle feature basata su z:
    [z, sin(wz), cos(wz), ..., sin(5wz), cos(5wz)]
    """
    if np.isscalar(z_arr):
        z_arr = np.array([z_arr])
    
    # Trend lineare z
    feats = [z_arr]
    
    # 5 Armoniche della frequenza fondamentale
    for k in range(1, 10):
        feats.append(np.sin(k * W_THEORETICAL * z_arr))
        feats.append(np.cos(k * W_THEORETICAL * z_arr))
        
    return np.column_stack(feats)

def reconstruct_targets(z_in):
    """
    Ricostruisce y1, y2, y3, y4 dato z usando le formule teoriche.
    """
    z = np.array(z_in)
    
    # Formule identificate
    y1 = 0.5463 * z * np.cos(1.1395 * z)
    y2 = 0.5463 * z * np.sin(1.1395 * z)
    
    # Sistema per y3, y4
    # y3 + y4 = -z * cos(2z)
    # y3 - y4 = z
    sum_y3y4 = -z * np.cos(2 * z)
    diff_y3y4 = z
    
    y3 = (sum_y3y4 + diff_y3y4) / 2.0
    y4 = (sum_y3y4 - diff_y3y4) / 2.0
    
    return np.column_stack([y1, y2, y3, y4])

def calculate_mee(y_true, y_pred):
    errors = y_true - y_pred
    dist = np.sqrt(np.sum(errors**2, axis=1))
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

# --- MAIN PROCESS ---

def main():
    print(f"Caricamento dati da {FILE_PATH}...")
    try:
        # Carica il CSV ignorando le righe di commento '#'
        df = pd.read_csv(FILE_PATH, comment='#', header=None)
    except FileNotFoundError:
        print("Errore: File non trovato. Controlla il percorso.")
        return

    # Estrazione colonne (ID è col 0, Input 1-12, Target 13-16)
    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values
    
    # Calcolo del vero z per il training (variabile latente)
    z_all = Y[:, 2] - Y[:, 3] # z = y3 - y4

    T, const, terms = parse_fit(FIT_PATH)

    random_state = 45
    print(f"random_state: {random_state}")
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    all_mae = []
    all_z_abs_err = []
    all_weighted_ssd = []
    all_z_var = []
    all_y_abs_err = []
    candidate_rows = []
    all_mee_before = []
    all_mee_after = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        z_train, z_val = z_all[train_idx], z_all[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        print(f"\nFold {fold_idx}:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_sc)
        X_val_pca = pca.transform(X_val_sc)
        pc2_train = X_train_pca[:, 1]
        pc2_val = X_val_pca[:, 1]

        # --- 1. ADDESTRAMENTO MODELLO DIRETTO (Forward Model x(z)) ---
        print("Addestramento modelli fisici per i 12 input...")

        models = []
        weights = []
        train_feats = get_physical_features(z_train)

        for i in range(12):
            # Fit Linear Regression su base armonica
            reg = LinearRegression().fit(train_feats, X_train[:, i])
            models.append(reg)

            # Calcolo MSE sul train per determinare l'affidabilità (Peso)
            preds = reg.predict(train_feats)
            mse = np.mean((X_train[:, i] - preds)**2)

            # Il peso è l'inverso dell'errore (più preciso = più peso)
            w = 1.0 / mse if mse > 1e-9 else 1.0
            weights.append(w)

        weights = np.array(weights)
        weights_norm = weights / np.sum(weights)

        print("Pesi assegnati alle variabili (Top 3):")
        top_idxs = np.argsort(weights)[::-1][:3]
        for idx in top_idxs:
            print(f"  x{idx+1}: {weights_norm[idx]:.4f}")

        # --- 2. PREPARAZIONE RICERCA INVERSA ---
        coefs_mat = np.array([m.coef_ for m in models])       # Shape (12, 11)
        intercepts = np.array([m.intercept_ for m in models]) # Shape (12,)

        z_min, z_max = z_train.min() - 5, z_train.max() + 5
        grid_size = 5000
        z_grid = np.linspace(z_min, z_max, grid_size)

        grid_feats = get_physical_features(z_grid) # (5000, 11)
        grid_preds = grid_feats @ coefs_mat.T + intercepts

        # --- 3. ESECUZIONE SU VALIDATION SET ---
        print("Esecuzione Ricerca Inversa Pesata sul Validation Set...")

        z_recovered = []
        z_candidates_all = []

        for i in range(len(X_val)):
            x_obs = X_val[i]

            diff_sq = (grid_preds - x_obs)**2
            weighted_ssd = np.sum(diff_sq * weights, axis=1)

            best_grid_idx = np.argmin(weighted_ssd)
            z_init = z_grid[best_grid_idx]
            all_weighted_ssd.append(float(weighted_ssd[best_grid_idx]))
            k = min(K_SSD, len(weighted_ssd))
            idx_k = np.argpartition(weighted_ssd, k - 1)[:k]
            z_candidates = z_grid[idx_k]
            z_var = float(np.var(z_candidates))
            all_z_var.append(z_var)
            z_candidates_all.append(z_candidates)

            z_recovered.append(z_init)

        z_recovered = np.array(z_recovered)

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

        # --- 4. VALUTAZIONE ---
        mae_z = np.mean(np.abs(z_recovered_refined - z_val))
        Y_pred_before = reconstruct_targets(z_recovered)
        mee_before = calculate_mee(Y_val, Y_pred_before)
        Y_pred_after = reconstruct_targets(z_recovered_refined)
        mee_after = calculate_mee(Y_val, Y_pred_after)
        y_abs_err = np.linalg.norm(Y_pred_after - Y_val, axis=1)

        print("RISULTATI FINALI:")
        print("-" * 30)
        print(f"MAE su z (recuperato): {mae_z:.4f}")
        print(f"MEE sui target (before refine): {mee_before:.4f}")
        print(f"MEE sui target (after refine):  {mee_after:.4f}")
        print("-" * 30)

        all_mae.append(mae_z)
        all_z_abs_err.extend(np.abs(z_recovered_refined - z_val))
        all_mee_before.append(mee_before)
        all_mee_after.append(mee_after)
        all_y_abs_err.extend(y_abs_err)
        for i in range(len(X_val)):
            candidates_str = ";".join([f"{val:.6f}" for val in z_candidates_all[i]])
            candidate_rows.append(
                {
                    "abs_z_err": float(np.abs(z_recovered_refined[i] - z_val[i])),
                    "z_true": float(z_val[i]),
                    "z_refined": float(z_recovered_refined[i]),
                    "z_candidates": candidates_str,
                }
            )

    print("\nRISULTATI FINALI (CV):")
    print("-" * 30)
    print(f"MAE su z (media): {np.mean(all_mae):.4f}")
    print(f"MEE sui target (before refine): {np.mean(all_mee_before):.4f}")
    print(f"MEE sui target (after refine):  {np.mean(all_mee_after):.4f}")
    print("-" * 30)

    out_dir = Path("plots/t5")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_abs_error_hist.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_z_abs_err, bins=40, color="tab:green", alpha=0.8)
    ax.set_xlabel("Absolute error |z_recovered - z|")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of absolute z errors (refined)")
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
    ax.set_ylabel("Absolute error |z_recovered - z|")
    ax.set_title("weighted_ssd vs absolute z error")
    fig.colorbar(sc, ax=ax, label="Var(z) for K lowest weighted_ssd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "abs_z_error_vs_abs_y_error.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_z_abs_err, all_y_abs_err, s=18, alpha=0.7, color="tab:blue")
    ax.set_xlabel("Absolute error |z_recovered - z|")
    ax.set_ylabel("Absolute error |y_recovered - y|")
    ax.set_title("Absolute z error vs absolute y error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    if candidate_rows:
        df_candidates = pd.DataFrame(candidate_rows)
        df_candidates = df_candidates.sort_values("abs_z_err", ascending=False)
        out_path = out_dir / "z_candidates_sorted_by_abs_err.csv"
        df_candidates.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")

if __name__ == "__main__":
    main()
