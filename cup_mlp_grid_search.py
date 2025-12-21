import itertools
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import load_cup_data 
# Importiamo la funzione di grid search dal tuo mlp.py aggiornato
from mlp import grid_search_kfold_cv

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
print("Caricamento dati CUP...")
X_cup, y_cup = load_cup_data('data/CUP/ML-CUP25-TR.csv')

# Preprocessing standard
scaler_x = StandardScaler()
X_cup = scaler_x.fit_transform(X_cup)

scaler_y = StandardScaler()
y_cup = scaler_y.fit_transform(y_cup)

# ==========================================
# 2. DEFINIZIONE GRIGLIA IPERPARAMETRI
# ==========================================
# Modifica questa griglia per esplorare diverse combinazioni.
# Ricorda: più opzioni metti, più tempo ci vorrà!
param_grid = {
    'hidden_layers': [
        [128, 512, 256, 128],   
        [128,  256, 128]
    ],
    'activation': ['ELU', 'LeakyReLU'],
    'lr': [0.005, 0.001], 
    'weight_decay': [1e-4], 
    'dropout': [0.0, 0.1],  
    'batch_size': [64],
    'epochs': [800],        
    'es': [True],           
    'patience': [50],       
    'loss': ['Huber', 'MSE'],
    'use_scheduler': [True],
    'momentum': [0.9]
}

# Generazione di tutte le combinazioni
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Totale combinazioni da testare per OGNI target: {len(combinations)}")
print("-" * 60)

# ==========================================
# 3. GRID SEARCH COMPONENTE PER COMPONENTE
# ==========================================
num_targets = y_cup.shape[1]
best_configs_per_target = []

for i in range(num_targets):
    print(f"\n🔵 INIZIO GRID SEARCH - TARGET {i+1} / {num_targets}")
    print("=" * 60)
    
    # Seleziona solo la colonna i-esima (mantieni 2D shape: N, 1)
    y_single = y_cup[:, i:i+1]
    
    # Lancia la Grid Search (5-Fold CV)
    # Nota: Task type 'regression' userà automaticamente il MEE (che su 1D è MAE/Euclidean)
    best_config, best_histories, best_avg_stop, all_results = grid_search_kfold_cv(
        combinations,
        k_folds=5,           # 5-Fold Cross Validation è robusta
        X=X_cup, 
        y=y_single,             
        task_type='regression'
    )
    
    # Trova il miglior score medio ottenuto
    best_val_score = min([res['mean_val_score'] for res in all_results] 
                         if 'mean_val_score' in all_results[0] else [0]) 
    
    # Se grid_search_kfold_cv stampa già, ottimo. Altrimenti recuperiamo qui.
    # (Il tuo mlp.py stampa durante l'esecuzione, quindi vedrai i log)
    
    print(f"\n🏆 VINCITORE TARGET {i+1}")
    print(f"Configurazione Migliore: {best_config}")
    # Nota: best_avg_stop ti dice a che epoca si è fermato in media (utile per settare epochs nell'ensemble)
    print(f"Stop Epoch Medio: {best_avg_stop:.1f}")
    
    best_configs_per_target.append(best_config)
    print("-" * 60)

# ==========================================
# 4. RIEPILOGO FINALE DA COPIARE
# ==========================================
print("\n\n" + "="*60)
print("RIEPILOGO CONFIGURAZIONI VINCENTI (Da copiare in cup_mlp.py)")
print("="*60)

for idx, conf in enumerate(best_configs_per_target):
    print(f"\n# Target {idx+1}")
    print(f"Config: {conf}")

print("\nCopia questi dizionari o i valori chiave (hidden_layers, lr, dropout) nello script 'cup_mlp.py' per l'Ensemble.")
