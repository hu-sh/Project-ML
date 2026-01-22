import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from utils import load_monk_data, get_encoder
import numpy as np
import matplotlib.pyplot as plt

from mlp import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SETUP DATI ---
k_folds = 5

j=1

train_path = 'data/MONK/monks-'+str(j)+'.train' 
X_raw_train, y_train = load_monk_data(train_path)
encoder = get_encoder(X_raw_train)
X_train_enc = encoder.transform(X_raw_train)

input_dim = X_train_enc.shape[1]

test_path = 'data/MONK/monks-'+str(j)+'.test' 
X_raw_test, y_test = load_monk_data(test_path)
encoder = get_encoder(X_raw_test)
X_test_enc = encoder.transform(X_raw_test)

X_test_t = torch.FloatTensor(X_test_enc).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

   

# --- 3. GRIGLIA IPERPARAMETRI ---
"""
best for monks1:
{'hidden_layers': [5], 'activation': 'tanh', 'lr': 0.3, 'momentum': 0.95, 'weight_decay': 8e-05, 'epochs': 300, 'batch_size': 32, 'es': False}
{'hidden_layers': [10], 'activation': 'ReLU', 'lr': 0.15, 'momentum': 0.95, 'weight_decay': 8e-05, 'epochs': 300, 'batch_size': 32, 'es': False
{'hidden_layers': [20], 'activation': 'ReLU', 'lr': 0.3, 'momentum': 0.95, 'weight_decay': 0.0001, 'epochs': 300, 'batch_size': 32, 'es': False}
"""
param_grid_monks1 = {
    'hidden_layers': [[20]], #[[5], [10], [20], [10, 10], [5,5,5]], 
    'activation': ["ReLU"],#["ReLU", "tanh"], 
    'lr': [0.3],#[0.1, 0.2, 0.3, 0.15, 0.07], 
    'momentum': [0.95], #[0.95, 0.9, 0.5, 0.2, 0.1], 
    'weight_decay': [0.0001],#[0.001,0.0001,0.00008, 0], 
    'epochs': [300], #[100, 150,300,500],
    'batch_size': [32],
    'es': [False]
}
param_grid_monks2 = {
    'hidden_layers': [[20]],#[[5], [10], [20], [10, 10], [5,5,5]], 
    'activation': ["ReLU"],#["ReLU", "tanh"], 
    'lr': [0.2],#[0.2, 0.3, 0.1, 0.05, 0.01], 
    'momentum': [0.95],#[0.95, 0.9, 0.5, 0.2, 0.1],
    'weight_decay': [0.00008],#[0.001,0.0001,0.00008,0], 
    'epochs': [150],#[150,100, 300, 500],
    'batch_size': [32],
    'es': [False]
}
param_grid_monks3 = {
    'hidden_layers': [[10]],#[[5], [10], [20], [10, 10], [5,5,5]], 
    'activation': ["ReLU"],#["ReLU", "tanh"], 
    'lr': [0.05],#[0.1, 0.05, 0.01], 
    'momentum': [0.9],#[0.9,0.5,0.2], 
    'weight_decay': [0.0001],#[0.0001, 0.001], 
    'epochs': [600],
    'batch_size': [32],
    'es': [True]
}


param_grid = param_grid_monks1 if j==1 else (param_grid_monks2 if j==2 else param_grid_monks3)

# Genera tutte le combinazioni
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Combinations to test: {len(combinations)}")

best_config, best_histories, best_avg_stop, all_results = grid_search_kfold_cv(combinations, k_folds, X_train_enc, y_train, task_type='classification')
df = pd.DataFrame(all_results)

avg_best_history = average_histories(best_histories)
plot_training_history(avg_best_history, task_type="classification")

print("\n-------------------------------------------")
print(f"🏆 Best Configuration Found (Acc: {avg_best_history['val_score'][-1]:.4f}):")
print(f"Hidden layers: {best_config['hidden_layers']}")
print(f"Activation function: {best_config['activation']}")
print(f"LR: {best_config['lr']} | Momentum: {best_config['momentum']}")
print(f"Weight decay: {best_config['weight_decay']}")
print(f"Best average stop: {best_avg_stop}")
print("-------------------------------------------")



############ BEST CONFIG TEST ON MONKS.TEST
X = torch.FloatTensor(X_train_enc)
y = torch.FloatTensor(y_train).view(-1, 1)
best_config['es'] = False
best_config['epochs'] = int(best_avg_stop) # you could have done early stopping with a 90/10 split
model, history, _ = train_model(best_config, input_dim, X, y, X_test_t, y_test_t, task_type='classification')
model.eval()
with torch.no_grad():
    preds = model(X_test_t)
    pred_lbl = (preds > 0.5).float()
    acc = (pred_lbl == y_test_t).sum() / len(y_test_t)
    acc_test = acc.item()
    mse_test = ((((preds - y_test_t)**2).sum())/ len(y_test_t)).item()
    print("Accuracy on test set: ", acc_test)
    print("Accuracy on TR: ", history['train_score'][-1])
    print("MSE TS/TR: ", mse_test, history['train_loss'][-1])


plot_training_history(history, task_type="classification", is_test = True)

