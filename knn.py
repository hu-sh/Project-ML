import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

# --- CONFIGURAZIONE ---
FILENAME = 'data/CUP/ML-CUP25-TR.csv'
N_PCA_COMPONENTS = 5  # Ottimale per il manifold
K_NEIGHBORS = 2       # Ottimale per la precisione locale
RIDGE_ALPHA = 0.0000001   # Regolarizzazione minima
FIT_PATH = 'fit.txt'
LAMBDA_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]


def _parse_fit(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    T = None
    const = 0.0
    terms = []
    for line in lines:
        if line.startswith('T ='):
            T = float(line.split('=', 1)[1].strip())
            continue
        if line.startswith('f(') or line.startswith('x ='):
            continue
        if re.fullmatch(r'[+-]?[0-9.eE+-]+', line):
            const += float(line)
            continue
        m = re.match(r'([+-])\s*([0-9.eE+-]+)\s*\*\s*(sin|cos)\((\d+)\*pi\*x/T\)', line)
        if not m:
            raise ValueError(f'Unrecognized line in fit: {line}')
        sign, coeff, kind, k = m.groups()
        coeff = float(coeff)
        if sign == '-':
            coeff = -coeff
        terms.append((coeff, kind, int(k)))
    if T is None:
        raise ValueError('Missing T in fit file')
    return T, const, terms


def _fit_eval(z, T, const, terms):
    out = const
    for coeff, kind, k in terms:
        arg = (k * np.pi * z) / T
        out += coeff * (np.sin(arg) if kind == 'sin' else np.cos(arg))
    return out


def _fit_deriv(z, T, terms):
    out = 0.0
    for coeff, kind, k in terms:
        arg = (k * np.pi * z) / T
        factor = (k * np.pi) / T
        if kind == 'sin':
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

def main():
    print("--- CUP 2025: Physics-Based Reconstruction Model ---")
    
    # 1. Caricamento Dati
    try:
        # skiprows=7 salta l'header testuale
        df = pd.read_csv(FILENAME, skiprows=7, header=None)
    except FileNotFoundError:
        # Fallback se il file è nella directory corrente
        try:
            df = pd.read_csv('ML-CUP25-TR.csv', skiprows=7, header=None)
        except:
            print("Errore: File dati non trovato.")
            return

    # Separazione Input (1-12) e Target (13-16)
    X = df.iloc[:, 1:13].values
    Y_full = df.iloc[:, 13:17].values

    # Calcoliamo il target latente z
    # z = y3 - y4
    z_target = (Y_full[:, 2] - Y_full[:, 3]).reshape(-1, 1)

    print(f"Dataset caricato: {X.shape[0]} campioni.")

    # 2. Split Training / Validation (80/20)
    # Importante: splittiamo anche Y_full per calcolare il MEE finale
    X_train, X_val, z_train, z_val, Y_train_full, Y_val_full = train_test_split(
        X, z_target, Y_full, test_size=0.2, random_state=47
    )

    # 3. Preprocessing
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # 4. Manifold Learning (PCA)
    # Usiamo la PCA solo per calcolare la distanza (trovare i vicini corretti)
    pca = PCA(n_components=N_PCA_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca = pca.transform(X_val_sc)

    # 5. Local Learning (Hybrid Approach)
    print(f"Addestramento Local Ridge (PCA={N_PCA_COMPONENTS}, K={K_NEIGHBORS})...")
    
    # Cerca i vicini nello spazio PCA
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(X_train_pca)
    distances, indices = nbrs.kneighbors(X_val_pca)

    z_preds = []
    for i in range(len(X_val)):
        idx = indices[i]
        
        # Prendi i vicini (Input originali scalati e Target z)
        X_loc = X_train_sc[idx]
        z_loc = z_train[idx]
        
        # Regressione Ridge Locale sulle feature ORIGINALI
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_loc, z_loc)
        
        # Predizione puntuale
        pred = model.predict([X_val_sc[i]])[0]
        z_preds.append(pred)

    z_pred_val = np.array(z_preds).flatten()

    # 6. Valutazione su z (Variabile Latente)
    mae_z = np.mean(np.abs(z_pred_val - z_val.flatten()))
    print(f"\nMAE sulla variabile latente z (y3-y4): {mae_z:.5f}")

    # 7. Ricostruzione Fisica dei Target
    print("Ricostruzione variabili y1, y2, y3, y4 dalle formule fisiche...")
    
    # Costanti fisiche
    k1 = 0.5463
    k2 = 1.1395
    
    z = z_pred_val
    
    # Formule dirette
    y1_rec = k1 * z * np.cos(k2 * z)
    y2_rec = k1 * z * np.sin(k2 * z)
    
    # Sistema inverso per y3, y4
    # Sappiamo: sum = y3+y4, diff = z
    # Relazione nota: sum = -diff * cos(2 * diff)
    sum_rec = -z * np.cos(2 * z)
    
    y3_rec = (sum_rec + z) / 2.0
    y4_rec = (sum_rec - z) / 2.0
    
    # Matrice ricostruita
    Y_pred_reconstructed = np.column_stack([y1_rec, y2_rec, y3_rec, y4_rec])

    # 8. Calcolo MEE Finale (baseline)
    # Confrontiamo la ricostruzione con i veri target originali (Y_val_full)
    diff = Y_pred_reconstructed - Y_val_full
    mee_final = np.mean(np.linalg.norm(diff, axis=1))

    print(f"\n--------------------------------------------------")
    print(f"MEE FINALE (Variabili Ricostruite - z KNN): {mee_final:.5f}")
    print(f"--------------------------------------------------")

    # 9. Refinement with pc2 fit + lambda sweep (select best MEE)
    try:
        T, const, terms = _parse_fit(FIT_PATH)
        z_min = float(z_target.min())
        z_max = float(z_target.max())

        pca2 = PCA(n_components=2, random_state=0)
        X_train_pca2 = pca2.fit_transform(X_train_sc)
        X_val_pca2 = pca2.transform(X_val_sc)
        pc2_val = X_val_pca2[:, 1]

        best = None
        for lam in LAMBDA_SWEEP:
            z_ref = np.array(
                [
                    _refine_z_reg(z0, pc2, T, const, terms, z_min, z_max, lam)
                    for z0, pc2 in zip(z_pred_val, pc2_val)
                ]
            )

            y1_rec = k1 * z_ref * np.cos(k2 * z_ref)
            y2_rec = k1 * z_ref * np.sin(k2 * z_ref)
            sum_rec = -z_ref * np.cos(2 * z_ref)
            y3_rec = (sum_rec + z_ref) / 2.0
            y4_rec = (sum_rec - z_ref) / 2.0
            Y_pred = np.column_stack([y1_rec, y2_rec, y3_rec, y4_rec])

            mee = np.mean(np.linalg.norm(Y_pred - Y_val_full, axis=1))
            if best is None or mee < best[0]:
                best = (mee, lam, z_ref, Y_pred)

        if best is not None:
            best_mee, best_lam, best_z, best_Y_pred = best
            mae_z_ref = np.mean(np.abs(best_z - z_val.flatten()))
            print(f"\nMiglior lambda (MEE): {best_lam} -> MEE: {best_mee:.5f}")
            print(f"MAE z (refined): {mae_z_ref:.5f}")
    except FileNotFoundError:
        print(f"\nFit file not found: {FIT_PATH}. Skipping refinement.")
    
    # Diagnostica Amplificazione
    print("\nNota Diagnostica:")
    if mee_final > 10 * mae_z:
        print(f"ATTENZIONE: Forte amplificazione dell'errore rilevata.")
        print(f"Un errore medio di {mae_z:.4f} su z è diventato {mee_final:.4f} sul target finale.")
        print(f"Fattore di amplificazione: {mee_final/mae_z:.2f}x")
    else:
        print("Amplificazione dell'errore sotto controllo.")

if __name__ == "__main__":
    main()
