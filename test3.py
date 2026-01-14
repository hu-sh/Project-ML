import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.optimize import differential_evolution, minimize_scalar

# --- CONFIGURAZIONE ---
FILENAME = 'data/CUP/ML-CUP25-TR.csv'
N_PCA_COMPONENTS = 4  # Numero di componenti PCA da pesare
K_NEIGHBORS = 2       # Numero di vicini per la regressione locale
RIDGE_ALPHA = 0.01   # Regolarizzazione per la Ridge locale

def calculate_mee(y_true, y_pred):
    """Calcola il Mean Euclidean Error."""
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))

def local_ridge_predict(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train):
    """
    Esegue la predizione locale pesando le componenti PCA.
    """
    # Applichiamo i pesi alle componenti PCA (radice per agire linearmente sulla distanza)
    W_sqrt = np.sqrt(np.maximum(weights, 0))
    X_train_weighted = X_train_pca * W_sqrt
    X_val_weighted = X_val_pca * W_sqrt
    
    # Ricerca dei vicini nello spazio PCA pesato
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(X_train_weighted)
    _, indices = nbrs.kneighbors(X_val_weighted)
    
    y_preds = []
    for i in range(len(X_val_sc)):
        idx = indices[i]
        # Ridge locale sui 12 input originali (scalati) per predire i 4 output
        X_loc = X_train_sc[idx]
        Y_loc = Y_train[idx]
        
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_loc, Y_loc)
        
        pred = model.predict(X_val_sc[i:i+1])
        y_preds.append(pred[0])
    
    return np.array(y_preds)

def objective_function(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train, Y_val):
    """Funzione obiettivo da minimizzare (MEE)."""
    y_pred = local_ridge_predict(weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train)
    return calculate_mee(Y_val, y_pred)

def main():
    print("--- CUP 2025: Optimization of PCA Weights for 4-Output Local Ridge ---")
    
    # 1. Caricamento Dati
    try:
        df = pd.read_csv(FILENAME, skiprows=7, header=None)
    except FileNotFoundError:
        print(f"Errore: File {FILENAME} non trovato.")
        return

    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values

    # 2. Split e Preprocessing (Standard per tutti i calcoli)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    pca = PCA(n_components=N_PCA_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca = pca.transform(X_val_sc)

    # 3. Ottimizzazione Evolutiva (Differential Evolution)
    # Definiamo i bound per i pesi delle 4 componenti PCA (es. tra 0 e 10)
    bounds = [(0, 10)] * N_PCA_COMPONENTS
    
    print("Avvio ottimizzazione dei pesi (questo potrebbe richiedere qualche minuto)...")
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train, Y_val),
        strategy='best1bin',
        maxiter=10,     # Aumentare per risultati più precisi
        popsize=15,      # Dimensione della popolazione
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )

    # 4. Risultati Finali
    best_weights = result.x
    min_mee = result.fun
    
    print("\n" + "="*50)
    print("OTTIMIZZAZIONE COMPLETATA")
    print(f"Migliori Pesi PCA trovati: {best_weights}")
    print(f"MEE minimo raggiunto su Val: {min_mee:.5f}")
    print("="*50)

    # Verifica finale con i pesi ottimizzati
    y_final_pred = local_ridge_predict(best_weights, X_train_sc, X_val_sc, X_train_pca, X_val_pca, Y_train)
    final_mee = calculate_mee(Y_val, y_final_pred)
    print(f"Validazione MEE Finale: {final_mee:.5f}")

    # Ricostruzione Y cercando z che minimizza ||y(z) - y_final_pred||
    z_min = (Y_train[:, 2] - Y_train[:, 3]).min() - 10
    z_max = (Y_train[:, 2] - Y_train[:, 3]).max() + 10
    k1 = 0.5463
    k2 = 1.1395

    def y_from_z(z_val):
        y1_rec = k1 * z_val * np.cos(k2 * z_val)
        y2_rec = k1 * z_val * np.sin(k2 * z_val)
        sum_rec = -z_val * np.cos(2 * z_val)
        y3_rec = (sum_rec + z_val) / 2.0
        y4_rec = (sum_rec - z_val) / 2.0
        return np.array([y1_rec, y2_rec, y3_rec, y4_rec])

    z_est = []
    y_from_z_list = []
    for y_hat in y_final_pred:
        z0 = y_hat[2] - y_hat[3]
        lo = max(z_min, z0 - 5.0)
        hi = min(z_max, z0 + 5.0)

        # Coarse grid to get a good start
        grid = np.linspace(lo, hi, 41)
        errs = [np.linalg.norm(y_from_z(zg) - y_hat) for zg in grid]
        z_start = grid[int(np.argmin(errs))]

        # Local refinement
        def obj(zg):
            return np.linalg.norm(y_from_z(zg) - y_hat)

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        z_best = res.x if res.success else z_start
        z_est.append(z_best)
        y_from_z_list.append(y_from_z(z_best))

    y_from_z = np.vstack(y_from_z_list)
    mee_from_z = calculate_mee(Y_val, y_from_z)
    print(f"MEE ricostruendo Y da z (y3-y4): {mee_from_z:.5f}")

if __name__ == "__main__":
    main()
