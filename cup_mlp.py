import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dai tuoi file locali
from utils import load_cup_data
from mlp import train_model, plot_training_history

# Configurazione Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. FUNZIONE DI FEATURE ENGINEERING
# ==============================================================================
def compute_derived_data(X_raw, y_raw):
    """
    Calcola feature e target.
    Restituisce anche i componenti per il Modello 3 (Direction).
    """
    # --- INPUTS BASE (Escludiamo x6 -> indice 5 per M1 e M2) ---
    indices_base = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    X_base = X_raw[:, indices_base]
    X_base_abs = np.abs(X_base)
    
    # Feature isolata x6
    x6_col = X_raw[:, 5].reshape(-1, 1)

    # --- TARGETS ---
    # A. Target Diff (y3 - y4)
    y_diff = (y_raw[:, 2] - y_raw[:, 3]).reshape(-1, 1)

    # B. Target Statistico: Media(|y1|..|y4|)
    # Useremo sum = 4 * mean per i calcoli
    y_mean_abs = np.mean(np.abs(y_raw), axis=1).reshape(-1, 1)
    
    # C. Target Direzione (y1, y2 normalizzati)
    # Norm L2 di y1, y2
    norm_12 = np.sqrt(y_raw[:, 0]**2 + y_raw[:, 1]**2).reshape(-1, 1)
    # Evitiamo divisioni per zero (anche se improbabile nel CUP)
    norm_12 = np.maximum(norm_12, 1e-6)
    
    u1 = (y_raw[:, 0] / norm_12[:, 0]).reshape(-1, 1)
    u2 = (y_raw[:, 1] / norm_12[:, 0]).reshape(-1, 1)
    y_direction = np.hstack([u1, u2])

    # D. Target Reale y2 (per verifica)
    y2_true = y_raw[:, 1]

    return X_base, X_base_abs, x6_col, y_diff, y_mean_abs, y_direction, y2_true

# ==============================================================================
# 2. CARICAMENTO E SPLIT
# ==============================================================================
print("\n" + "="*50)
print("FASE 1: CARICAMENTO E SPLIT DATI")
print("="*50)

X_full, y_full = load_cup_data('data/CUP/ML-CUP25-TR.csv')

# Split: Test (10%), Train (72%), Val (18%)
X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(X_full, y_full, test_size=0.1, random_state=42)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Calcolo dati derivati
# Train
X_tr, X_tr_abs, x6_tr, y_tr_diff, y_tr_mean, y_tr_dir, _ = compute_derived_data(X_train_raw, y_train_raw)
# Val
X_vl, X_vl_abs, x6_vl, y_vl_diff, y_vl_mean, y_vl_dir, _ = compute_derived_data(X_val_raw, y_val_raw)
# Test
X_ts, X_ts_abs, x6_ts, y_ts_diff, y_ts_mean, y_ts_dir, y_ts_target_y2 = compute_derived_data(X_test_raw, y_test_raw)

# ==============================================================================
# 3. PREPARAZIONE INPUT PER MODELLO 3 (Direction)
# ==============================================================================
# L'input per il modello 3 è: [y3-y4, x6, abs(y1)+abs(y2)]
# abs(y1)+abs(y2) approssimato come: (4 * mean_abs) - abs(y3-y4)

def build_m3_input(diff_val, mean_val, x6_val):
    sum_total = 4 * mean_val
    # La feature richiesta: sum(abs(y_j)) - abs(y3-y4)
    feat_sum_12 = sum_total - np.abs(diff_val)
    return np.column_stack([diff_val, x6_val, feat_sum_12])

# Costruiamo gli input di training usando i TARGET VERI (Teacher Forcing)
X_tr_m3 = build_m3_input(y_tr_diff, y_tr_mean, x6_tr)
X_vl_m3 = build_m3_input(y_vl_diff, y_vl_mean, x6_vl)
# Nota: Per X_ts_m3 lo faremo dopo usando le PREDIZIONI, non i target veri!

# ==============================================================================
# 4. SCALING
# ==============================================================================
print("\n" + "="*50)
print("FASE 2: SCALING (Solo Input)")
print("="*50)

# M1 & M2 Scaler
scaler_base = StandardScaler()
X_tr_std = scaler_base.fit_transform(X_tr)
X_vl_std = scaler_base.transform(X_vl)
X_ts_std = scaler_base.transform(X_ts)

scaler_abs = StandardScaler()
X_tr_abs_std = scaler_abs.fit_transform(X_tr_abs)
X_vl_abs_std = scaler_abs.transform(X_vl_abs)
X_ts_abs_std = scaler_abs.transform(X_ts_abs)

# M3 Scaler (Input composti)
scaler_m3 = StandardScaler()
X_tr_m3_std = scaler_m3.fit_transform(X_tr_m3)
X_vl_m3_std = scaler_m3.transform(X_vl_m3)

# Target: NESSUNO SCALING
# y_tr_diff, y_tr_mean, y_tr_dir restano grezzi.
# y_tr_dir sono valori tra -1 e 1 (cos/sin), non serve scaling.

# ==============================================================================
# 5. TRAINING
# ==============================================================================
def to_tensor(arr):
    return torch.FloatTensor(arr).to(device)

# Dataset Tensor M1/M2
X_tr_t = to_tensor(X_tr_std); y_tr_diff_t = to_tensor(y_tr_diff)
X_vl_t = to_tensor(X_vl_std); y_vl_diff_t = to_tensor(y_vl_diff)

X_tr_abs_t = to_tensor(X_tr_abs_std); y_tr_mean_t = to_tensor(y_tr_mean)
X_vl_abs_t = to_tensor(X_vl_abs_std); y_vl_mean_t = to_tensor(y_vl_mean)

# Dataset Tensor M3
X_tr_m3_t = to_tensor(X_tr_m3_std); y_tr_dir_t = to_tensor(y_tr_dir)
X_vl_m3_t = to_tensor(X_vl_m3_std); y_vl_dir_t = to_tensor(y_vl_dir)

config = {
    'hidden_layers': [],
    'activation': 'tanh',
    'lr': 0.001, 
    'epochs': 10000, 
    'batch_size': 32, 
    'patience': 50,
    'dropout': 0,
    'es': True
}

print("\n--- Training Model 1: Diff (y3-y4) ---")
#config['hidden_layers'] = []
model_diff, hist_diff, _ = train_model(
    config, X_tr_std.shape[1], X_tr_t, y_tr_diff_t, X_vl_t, y_vl_diff_t
)
print(f"Valid MEE: {hist_diff['val_score'][-1]:.5f}")
#print(model_diff.cpu().net[0].weight)
#model_diff.to(device)




print("\n--- Training Model 2: Mean Abs ---")
config['hidden_layers'] = [12]
model_mean, hist_mean, _ = train_model(
    config, X_tr_abs_std.shape[1], X_tr_abs_t, y_tr_mean_t, X_vl_abs_t, y_vl_mean_t
)
print(f"Valid MEE: {hist_mean['val_score'][-1]:.5f}")


# ==============================================================================
# 6. TRAINING MODEL 4 (SOLVER) - Basato su cup_mlp_old
# ==============================================================================
from sklearn.preprocessing import PolynomialFeatures

print("\n" + "="*50)
print("FASE 3: ADDESTRAMENTO SOLVER SULLE PREDIZIONI DI M1")
print("="*50)

# --- A. Generazione Input: Predizioni di M1 (y3-y4) ---
# Usiamo il modello appena addestrato per generare gli input "sporchi" per il solver
model_diff.eval()
with torch.no_grad():
    # Otteniamo le predizioni per train e validation
    y_tr_diff_pred = model_diff(X_tr_t).cpu().numpy()
    y_vl_diff_pred = model_diff(X_vl_t).cpu().numpy()

# --- B. Feature Engineering: Espansione Polinomiale ---
# Come in cup_mlp_old, espandiamo la singola feature (la diff) in grado 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_solver_poly = poly.fit_transform(y_tr_diff_pred)
X_vl_solver_poly = poly.transform(y_vl_diff)

# --- C. Scaling degli input polinomiali ---

# --- D. Definizione Target: y1, y2, y4 ---
# Il Solver impara a mappare la differenza verso gli altri 3 target
y_tr_solver_target = y_train_raw[:, [0, 1, 3]]
y_vl_solver_target = y_val_raw[:, [0, 1, 3]]

# Conversione in Tensor per la tua libreria
X_tr_solver_t = to_tensor(X_tr_solver_poly)
y_tr_solver_t = to_tensor(y_tr_solver_target)
X_vl_solver_t = to_tensor(X_vl_solver_poly)
y_vl_solver_t = to_tensor(y_vl_solver_target)

# --- E. Training con DynamicMLP ---
# Configurazione che ricalca SolverNet usando i parametri di mlp.py
config_solver = {
    'hidden_layers': [256, 512, 256, 128], # Architettura profonda come in cup_mlp_old
    'activation': 'ELU',     # ELU è ottima per la regressione smooth
    'lr': 0.001,
    'epochs': 2000,
    'batch_size': 64,
    'patience': 100,
    'optim': 'adam',         # Adam per convergenza rapida
    'loss': 'Huber',         # HuberLoss per gestire meglio l'errore delle predizioni
    'es': True
}

print("\n--- Training Model 4: Solver (y3-y4_pred -> y1, y2, y4) ---")
model_solver, hist_solver, stop_epoch_s = train_model(
    config_solver, 
    X_tr_solver_poly.shape[1], 
    X_tr_solver_t, 
    y_tr_solver_t, 
    X_vl_solver_t, 
    y_vl_solver_t
)

# --- F. Funzione di Ricostruzione Finale ---
def get_final_targets(diff_preds, model_s, poly_s):
    """
    Data la predizione di y3-y4, calcola il vettore completo [y1, y2, y3, y4]
    """
    model_s.eval()
    with torch.no_grad():
        X_p = poly_s.transform(diff_preds)
        X_s = X_p
        X_t = to_tensor(X_s)
        
        # Predizione di y1, y2, y4
        out = model_s(X_t).cpu().numpy()
        y1, y2, y4 = out[:, 0], out[:, 1], out[:, 2]
        
        # Ricostruzione algebrica: y3 = y4 + (y3-y4)
        y3 = y4 + diff_preds.flatten()
        
        return np.column_stack([y1, y2, y3, y4])

# Valutazione finale sul Validation Set
final_preds_vl = get_final_targets(y_vl_diff_pred, model_solver, poly)
final_mee = np.mean(np.linalg.norm(final_preds_vl - y_val_raw, axis=1))

print(f"\nVALIDATION MEE FINALE (Catena M1 + Solver): {final_mee:.5f}")

