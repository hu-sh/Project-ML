import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import sys

FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
W_THEORETICAL = 0.57#(1 + 1/np.sqrt(2)) / 3 

def get_physical_features(z_arr):
    """
    Costruisce la matrice delle feature basata su z:
    [z, sin(wz), cos(wz), ..., sin(5wz), cos(5wz)]
    """
    if np.isscalar(z_arr):
        z_arr = np.array([z_arr])
    
    feats = [z_arr]
    
    for k in range(1, 7):
        feats.append(np.sin(k * W_THEORETICAL * z_arr))
        feats.append(np.cos(k * W_THEORETICAL * z_arr))
        
    return np.column_stack(feats)

def reconstruct_targets(z_in):
    """
    Ricostruisce y1, y2, y3, y4 dato z usando le formule teoriche.
    """
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
    dist = np.sqrt(np.sum(errors**2, axis=1))
    return np.mean(dist)


def main():
    print(f"Caricamento dati da {FILE_PATH}...")
    try:
        # Carica il CSV ignorando le righe di commento '#'
        df = pd.read_csv(FILE_PATH, comment='#', header=None)
    except FileNotFoundError:
        print("Errore: File non trovato. Controlla il percorso.")
        return

    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values
    
    z_all = Y[:, 2] - Y[:, 3] # z = y3 - y4

    X_train, X_val, z_train, z_val, Y_train, Y_val = train_test_split(
        X, z_all, Y, test_size=0.15#, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    print("\nAddestramento modelli fisici per i 12 input...")
    
    models = []
    weights = []
    train_feats = get_physical_features(z_train)
    
    for i in range(12):
        reg = LinearRegression().fit(train_feats, X_train[:, i])
        models.append(reg)
        
        preds = reg.predict(train_feats)
        mse = np.mean((X_train[:, i] - preds)**2)
        
        w = 1.0 / mse if mse > 1e-9 else 1.0
        weights.append(w)
    
    weights = np.array(weights)
    weights_norm = weights / np.sum(weights)
    
    print("Pesi assegnati alle variabili (Top 3):")
    top_idxs = np.argsort(weights)[::-1][:3]
    for idx in top_idxs:
        print(f"  x{idx+1}: {weights_norm[idx]:.4f}")

    # --- 2. PREPARAZIONE RICERCA INVERSA ---
    # Matrici per calcolo vettoriale
    coefs_mat = np.array([m.coef_ for m in models])       
    intercepts = np.array([m.intercept_ for m in models])
    
    # Creazione Griglia di Ricerca (Grid Search)
    # Estendiamo leggermente il range osservato nel training
    z_min, z_max = z_train.min() - 5, z_train.max() + 5
    grid_size = 50000
    z_grid = np.linspace(z_min, z_max, grid_size)
    
    # Pre-calcolo delle predizioni su tutta la griglia (Vectorized)
    # Questo velocizza enormemente il processo rispetto a farlo nel loop
    grid_feats = get_physical_features(z_grid) # (5000, 11)
    # Predizioni teoriche per ogni punto della griglia: (5000, 12)
    grid_preds = grid_feats @ coefs_mat.T + intercepts
    
    # --- 3. ESECUZIONE SU VALIDATION SET ---
    print("\nEsecuzione Ricerca Inversa Pesata sul Validation Set...")
    
    z_recovered = []
    
    for i in range(len(X_val)):
        x_obs = X_val[i] # L'input osservato (12,)
        
        # A. GRID SEARCH (Trova il minimo globale approssimato)
        # Calcolo differenza quadratica pesata per ogni punto della griglia
        # (5000, 12) - (12,) -> broadcast
        diff_sq = (grid_preds - x_obs)**2
        # Somma pesata lungo le variabili: (5000,)
        weighted_ssd = np.sum(diff_sq * weights, axis=1)
        
        best_grid_idx = np.argmin(weighted_ssd)
        z_init = z_grid[best_grid_idx]
        
        # B. RAFFINAMENTO (Ottimizzazione locale fine)
        # Funzione obiettivo da minimizzare per il singolo punto
        def objective(z_curr):
            # Ricostruisce features per un singolo z
            f = get_physical_features(z_curr).flatten()
            # Predice x
            x_pred = coefs_mat.dot(f) + intercepts
            # Errore pesato
            return np.sum(((x_obs - x_pred)**2) * weights)
        
        # Minimize usando L-BFGS-B in un intorno del punto trovato
        res = minimize(objective, z_init, 
                       bounds=[(z_init - 1.0, z_init + 1.0)], 
                       method='L-BFGS-B')
        
        z_recovered.append(res.x[0])
    
    z_recovered = np.array(z_recovered)
    
    # --- 4. VALUTAZIONE ---
    
    # A. Errore su Z
    mae_z = np.mean(np.abs(z_recovered - z_val))
    
    # B. Ricostruzione Target Y e calcolo MEE
    Y_pred = reconstruct_targets(z_recovered)
    mee = calculate_mee(Y_val, Y_pred)
    
    print("\nRISULTATI FINALI:")
    print("-" * 30)
    print(f"MAE su z (recuperato): {mae_z:.4f}")
    print(f"MEE sui target (y):   {mee:.4f}")
    print("-" * 30)
    
    # Esempio di predizione vs realtà per il primo campione di validation
    print("\nEsempio (Primo campione di Validation):")
    print(f"Z Vero: {z_val[0]:.4f} | Z Stimato: {z_recovered[0]:.4f}")
    print(f"Y1 Vero: {Y_val[0][0]:.4f} | Y1 Stimato: {Y_pred[0][0]:.4f}")

if __name__ == "__main__":
    main()
