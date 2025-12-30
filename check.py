import pandas as pd
import numpy as np
import torch
import itertools
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

from utils import *
from mlp import grid_search_kfold_cv, average_histories 

X_raw, y_raw = load_cup_data("data/CUP/ML-CUP25-TR.csv") 
y_raw = y_raw[:,2]-y_raw[:,3]
y_raw = y_raw.reshape(-1, 1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. DEFINIZIONE DELLO SPAZIO DI RICERCA
# Parametri di Preprocessing (Gradi di libertà richiesti)
pca_k_options = [1,2,3,4,5, 8, 10]
poly_degree_options = [1, 2,4,8]

# Parametri MLP (Esempio di griglia)
mlp_grid = {
    'hidden_layers': [[], [128], [32, 32,32,32,32]],
    'activation': ['ReLU'],
    'lr': [0.01, 0.001],
    'wd': [0.1],
    #'dropout': [0.1],
    'epochs': [2000],
    'batch_size': [32],
    'optim': ['adam'],
    'loss': ['MSE'],
    'es': [True]
}

keys, values = zip(*mlp_grid.items())
mlp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


results_summary = []

# 4. LOOP ESAUSTIVO (Preprocessing + MLP)
for k in pca_k_options:
    for degree in poly_degree_options:
        
        # --- APPLICAZIONE PREPROCESSING ---
        # PCA
        pca = PCA(n_components=k)
        X_pca = pca.fit_transform(X_scaled)
        
        # Combinazioni Polinomiali
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_transformed = poly.fit_transform(X_pca)
        
        print(f"\n>>> Testando configurazione Preprocessing: PCA k={k}, Poly Degree={degree}")
        print(f"Dimensione input risultante: {X_transformed.shape[1]}")
        
        
        best_config, histories, avg_stop, results = grid_search_kfold_cv(
            combinations=mlp_combinations, 
            k_folds=5, 
            X=X_transformed, 
            y=y_raw, 
            task_type='regression',
            standardize=False
        )
        
        avg_bst_hist = average_histories(histories)
        print("avg-best-MEE: ", avg_bst_hist['val_score'][-1])
        
        # Salvataggio del risultato medio (calcolato dalla tua funzione)
        avg_mee = results[0]['avg_history']['val_score'][-1]
        

# 5. FINAL BEST
best_overall = min(results_summary, key=lambda x: x['avg_val_score'])
print("\n" + "="*50)
print("GRID SEARCH COMPLETATA")
print(f"Miglior Configurazione: {best_overall['config']}")
print(f"Miglior Val Score (MEE) medio: {best_overall['avg_val_score']:.6f}")
print("="*50)
