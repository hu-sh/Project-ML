import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- 1. CARICAMENTO ---
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', comment='#', header=None)
X_raw = df.values[:, 1:13].astype(np.float64)
Y_raw = df.values[:, 13:].astype(np.float64)
z_raw = Y_raw[:, 2] - Y_raw[:, 3]

# --- 2. FEATURE UNROLLING (Nyström) ---
# Questo trasforma gli input in modo da "sciogliere" le curvature non lineari
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw)

# n_components=600 crea 600 nuove feature che "mappano" la non-linearità
feature_map = Nystroem(kernel='rbf', gamma=0.1, n_components=600, random_state=42)
X_unrolled = feature_map.fit_transform(X_norm)

# --- 3. SPLIT (FIXATO: 6 variabili per 3 array) ---
# Passiamo: X_unrolled, z_raw, Y_raw -> Ritornano 6 valori
X_train, X_val, z_train, z_val, Y_train_val, Y_val = train_test_split(
    X_unrolled, z_raw, Y_raw, test_size=0.2, random_state=42
)

# --- 4. REGRESSIONE ROBUSTA ---
# Ridge su spazio Nyström è quasi potente come un SVR ma più stabile
model_z = Ridge(alpha=0.1)
model_z.fit(X_train, z_train)

# --- 5. RICOSTRUZIONE ---
z_pred = model_z.predict(X_val)

def reconstruct(z):
    # Formule originali
    y1 = 0.5463 * z * np.cos(1.1395 * z)
    y2 = 0.5463 * z * np.sin(1.1395 * z)
    sum_y = -z * np.cos(2.0 * z)
    y3 = (sum_y + z) / 2
    y4 = (sum_y - z) / 2
    return np.stack([y1, y2, y3, y4], axis=1)

Y_pred = reconstruct(z_pred)
mae_z = np.mean(np.abs(z_pred - z_val))
mee = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))

print(f"MAE su z: {mae_z:.4f}")
print(f"Val MEE: {mee:.4f}")
