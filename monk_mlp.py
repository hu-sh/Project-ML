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

train_path = 'data/MONK/monks-3.train' 
X_raw_train, y_train = load_monk_data(train_path)
encoder = get_encoder(X_raw_train)
X_train_enc = encoder.transform(X_raw_train)

input_dim = X_train_enc.shape[1]

test_path = 'data/MONK/monks-3.test' 
X_raw_test, y_test = load_monk_data(test_path)
encoder = get_encoder(X_raw_test)
X_test_enc = encoder.transform(X_raw_test)

X_test_t = torch.FloatTensor(X_test_enc).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

   

# --- 3. GRIGLIA IPERPARAMETRI ---
param_grid = {
    'hidden_layers': [[5], [10], [20], [10, 10]], 
    'activation': ["ReLU", "tanh"], 
    'lr': [0.1, 0.05], 
    'momentum': [ 0.9], 
    'weight_decay': [0.0001, 0.001], 
    'epochs': [300],
    'batch_size': [32],
    'es': [True]
}

# Genera tutte le combinazioni
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Combinations to test: {len(combinations)}")

best_config, best_histories, best_avg_stop, all_results = grid_search_kfold_cv(combinations, k_folds, X_train_enc, y_train, task_type='classification')
df = pd.DataFrame(all_results)

avg_best_history = average_histories(best_histories)
plot_training_history(avg_best_history)

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
    print("Accuracy on test set: ", acc_test)


plot_training_history(history)

