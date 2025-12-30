import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
from mlp import grid_search_kfold_cv, average_histories 

X_raw, y_raw = load_cup_data("data/CUP/ML-CUP25-TR.csv") 

# --- 1. PREPARAZIONE TARGET d ---
# Calcoliamo d (la differenza) che è il nostro unico grado di libertà
d_raw = y_raw[:, 2] - y_raw[:, 3] 

X_train, X_test, d_train, d_test, y_train_full, y_test_full = train_test_split(
    X_raw, d_raw, y_raw, test_size=0.2, random_state=42
)

# --- 2. ENSEMBLE DI 10 GBRT ---
# Definiamo il GBRT base con parametri robusti
base_gbrt = GradientBoostingRegressor(
    n_estimators=200,    # numero di alberi per ogni GBRT
    learning_rate=0.1,
    max_depth=5,         # profondità per catturare non-linearità
    min_samples_split=5,
    random_state=42
)

# Creiamo l'ensemble di 10 modelli (Bagging)
ensemble_model = BaggingRegressor(
    estimator=base_gbrt,
    n_estimators=10,     # numero di GBRT nell'ensemble
    max_samples=0.8,     # ogni modello vede l'80% dei dati
    n_jobs=-1,           # usa tutti i core della CPU
    random_state=42
)

print("Addestramento ensemble in corso...")
ensemble_model.fit(X_train, d_train)

# --- 3. PREDIZIONE DI d ---
d_pred = ensemble_model.predict(X_test)

# --- 4. RICOSTRUZIONE ANALITICA DI y1, y2, y3, y4 ---
# Usiamo le tue formule per "forzare" la geometria corretta
def reconstruct_targets(d):
    # y1 = 0.5463 * d * cos(1.1395 * d)
    y1_rec = 0.5463 * d * np.cos(1.1395 * d)
    # y2 = 0.5463 * d * sin(1.1395 * d)
    y2_rec = 0.5463 * d * np.sin(1.1395 * d)
    # y3 + y4 = -d * cos(2 * d) -> Sappiamo anche d = y3 - y4
    # Risolvendo il sistema:
    # y3 = ( (y3+y4) + (y3-y4) ) / 2
    # y4 = ( (y3+y4) - (y3-y4) ) / 2
    sum_y3y4 = -d * np.cos(2 * d)
    y3_rec = (sum_y3y4 + d) / 2
    y4_rec = (sum_y3y4 - d) / 2
    
    return np.column_stack([y1_rec, y2_rec, y3_rec, y4_rec])

y_pred_final = reconstruct_targets(d_pred)

# --- 5. CALCOLO MEE ---
def calculate_mee(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))

mee_val = calculate_mee(y_test_full, y_pred_final)
print(f"MEE Finale con Ensemble di GBRT: {mee_val:.4f}")
