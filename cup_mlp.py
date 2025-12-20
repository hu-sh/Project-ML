import itertools
import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import StandardScaler

from utils import load_cup_data 
from mlp import grid_search_kfold_cv, plot_training_history

X_cup, y_cup = load_cup_data('data/CUP/ML-CUP25-TR.csv')
scaler_x = StandardScaler()
X_cup = scaler_x.fit_transform(X_cup)

scaler_y = StandardScaler()
y_cup = scaler_y.fit_transform(y_cup)

X_cup = X_cup[:20]
y_cup = y_cup[:20]


param_grid_big = {
    'hidden_layers': [
        [512],              # Quella attuale (Benchmark)
        [1024],             # The "Tank": Larghissima, cattura tutto subito
        [512, 512],         # Deep & Wide: Capacità di astrazione maggiore
        [1024, 512],        # Imbuto grande
        [512, 256, 128]     # Imbuto classico profondo
    ],
    
    # LeakyReLU è la tua nuova migliore amica, non cambiarla.
    'activation': ['LeakyReLU'],
    
    # Ora che la rete è grande, potremmo rischiare l'overfitting.
    # Testiamo "zero freni" vs "frenata leggera".
    'weight_decay': [0.0, 1e-4], 
    
    'lr': [0.001], # Adam a 0.001 è quasi sempre giusto
    'epochs': [800],
    'batch_size': [64]
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Totale combinazioni da testare: {len(combinations)}")

# 3. Lancia Grid Search (is_regression=True è FONDAMENTALE)
best_config, best_hist, best_avg_stop, all_results = grid_search_kfold_cv(
    combinations,
    5,
    X_cup, 
    y_cup,             
    task_type='regression'   # <--- Dice alla funzione di minimizzare il MEE
)
# 4. Risultati
print("\n-------------------------------------------")
print(f"🏆 Miglior Configurazione CUP (MEE: {best_mee:.4f} +/- {best_std:.4f}):")
print(best_config)
print("-------------------------------------------")


# 6. Plotta la curva del modello vincente (di uno dei fold)
plot_training_history(best_hist, is_regression=True, title="Best CUP Config (Single Fold)")
