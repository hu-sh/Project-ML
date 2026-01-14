import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

# ==========================================
# 1. PARAMETRI E CARICAMENTO
# ==========================================
FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
N_COMPONENTS = 6  # <--- PARAMETRO DA MODIFICARE (es. 4, 6, 8, 12)

# Caricamento
df = pd.read_csv(FILE_PATH, comment='#', header=None, index_col=0)
X = df.iloc[:, 0:12].values
Y = df.iloc[:, 12:16].values 
z = Y[:, 2] - Y[:, 3] # z = y3 - y4 (Generatore Latente)

# Split Training/Validation
X_train, X_val, Y_train, Y_val, z_train, z_val = train_test_split(X, Y, z, test_size=0.1)

# ==========================================
# 2. PIPELINE: SCALER + PCA
# ==========================================
# Standardizzazione (essenziale per PCA)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

# PCA
pca = PCA(n_components=N_COMPONENTS)
X_train_pca = pca.fit_transform(X_train_scaled)
# Trasformiamo anche il validation set nello spazio PCA ridotto
X_val_pca = pca.transform(X_val_scaled)

print(f"--- SETUP PCA ---")
print(f"Componenti utilizzate: {N_COMPONENTS}")
print(f"Varianza Spiegata Cumulativa: {np.sum(pca.explained_variance_ratio_):.4%}")

# ==========================================
# 3. MODELLO FORWARD (z -> PCA_X)
# ==========================================
print("\nAddestramento Random Forest (z -> PCA features)...")
# Sostituisci questa riga:
model_rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
#model_rf = ExtraTreesRegressor(n_estimators=2000, n_jobs=-1, random_state=42)

model_rf.fit(z_train.reshape(-1, 1), X_train_pca)

# ==========================================
# 4. INVERSIONE (Ricerca nello spazio PCA)
# ==========================================
# Creazione Lookup Table
# Generiamo 50.000 possibili valori di z e vediamo cosa predice il modello (in spazio PCA)
z_min, z_max = z_train.min() - 10, z_train.max() + 10
z_grid = np.linspace(z_min, z_max, 500000)
X_grid_pca_pred = model_rf.predict(z_grid.reshape(-1, 1))

def predict_z_with_pca(input_raw):
    """
    1. Prende un input raw (12 dim).
    2. Lo trasforma in PCA (N dim).
    3. Cerca il vicino più prossimo nella lookup table PCA.
    4. Restituisce z corrispondente.
    """
    # Trasformazione Input -> PCA
    x_scaled = scaler_X.transform([input_raw])
    x_pca = pca.transform(x_scaled)
    
    # Calcolo Distanze (nello spazio ridotto PCA)
    # axis=1 calcola la norma per ogni riga
    dists = np.linalg.norm(X_grid_pca_pred - x_pca, axis=1)
    
    # Troviamo lo z che minimizza la distanza
    best_idx = np.argmin(dists)
    return z_grid[best_idx]

# ==========================================
# 5. RICOSTRUZIONE Y (Formule Analitiche)
# ==========================================
def reconstruct_y_from_z(z_val):
    # Formule geometriche note
    y1 = 0.5463 * z_val * np.cos(1.1395 * z_val)
    y2 = 0.5463 * z_val * np.sin(1.1395 * z_val)
    
    sum_y34 = -z_val * np.cos(2 * z_val)
    diff_y34 = z_val
    
    y3 = (sum_y34 + diff_y34) / 2
    y4 = (sum_y34 - diff_y34) / 2
    return np.array([y1, y2, y3, y4])

# ==========================================
# 6. VALUTAZIONE MEE
# ==========================================
print("\nValutazione MEE sul Validation Set...")

errors = []
z_errors = []
for i in range(len(X_val)):
    x_in = X_val[i]
    y_true = Y_val[i]
    
    # A. Stima z (passando per PCA)
    z_est = predict_z_with_pca(x_in)
    z_errors.append(abs(z_est - z_val[i]))
    
    # B. Ricostruzione Y
    y_est = reconstruct_y_from_z(z_est)
    
    # Errore Euclideo
    err = np.linalg.norm(y_est - y_true)
    errors.append(err)

mee = np.mean(errors)
print(f"MEE Totale: {mee:.5f}")

# Histogram of absolute errors for z_est
out_path = Path("plots/z_est_abs_error_hist.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(z_errors, bins=40, color="tab:orange", alpha=0.8)
ax.set_xlabel("Absolute error |z_est - z_val|")
ax.set_ylabel("Count")
ax.set_title("Histogram of absolute errors (z_est)")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved plot: {out_path}")
