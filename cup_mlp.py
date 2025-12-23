import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Importiamo le funzioni dai tuoi file
from utils import load_cup_data
from mlp import train_model, plot_training_history, device

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(X, y):
    # 1. Caricamento dati originali
        
    # 2. Target: sqrt(y1^2 + y2^2)
    y_custom = np.sqrt(y[:, 0]**2 + y[:, 1]**2).reshape(-1, 1)
    
    # 3. Feature Engineering: SOLO i valori assoluti dei prodotti pairwise
    # Rimuoviamo le variabili singole originali
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    products_list = []
    
    # Calcoliamo |xi * xj| per ogni coppia (inclusi i quadrati |xi * xi|)
    for i in range(num_features):
        for j in range(i, num_features):
            abs_product = np.abs(X[:, i] * X[:, j]).reshape(-1, 1)
            products_list.append(abs_product)
            
    # Concateniamo tutte le nuove feature in un unico array
    X_transformed = np.concatenate(products_list, axis=1)
    
    print(f"Shape finale (solo prodotti assoluti): {X_transformed.shape}") # Dovrebbe essere (500, 78)
    
    return X_transformed, y_custom

# --- MAIN EXECUTION ---

# Caricamento e trasformazione
X, y = load_cup_data('data/CUP/ML-CUP25-TR.csv')
#X_exp, y_target = prepare_data(X, y)

y_norm12 = np.sqrt(y[:, 0]**2 + y[:, 1]**2).reshape(-1, 1)
input_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
X_exp = np.abs(X[:, input_indices])


# Split in Training e Validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_exp, y_norm12, test_size=0.2, random_state=42
)


scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)

# Scaling del Target (y)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
# Scaliamo anche y_test per il calcolo della loss interna (val_loss)
y_val = scaler_y.transform(y_val)



# Conversione in Tensor per PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)



# Configurazione del modello MLP
config = {
    'hidden_layers': [64, 32], # Architettura esempio
    'activation': 'ReLU',
    'lr': 0.0001,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'epochs': 2000,
    'batch_size': 32,
    'optim': 'sgd',
    'use_scheduler': True,
    'loss': 'MSE',
    'es': True,         # Early Stopping attivo
    'patience': 30
}

# Addestramento
print("\nInizio training...")
model, history, stop_epoch = train_model(
    config, 
    input_dim=X_exp.shape[1], 
    X_train=X_train_t, 
    y_train=y_train_t, 
    X_val=X_val_t, 
    y_val=y_val_t, 
    task_type='regression'
)

# Plot dei risultati
plot_training_history(history, title="MLP: Predict sqrt(y1^2 + y2^2)")

print(f"\nTraining terminato all'epoca: {stop_epoch}")
print(f"MEE finale su Validation: {history['val_score'][-1]:.4f}")

# ===================================================
print("\n" + "="*40)
print("INIZIO ANALISI PER: y3 - y4")
print("="*40)

# 1. Definizione del Target: Differenza y3 - y4
# y[:, 2] è y3, y[:, 3] è y4
y_diff34 = (y[:, 2] - y[:, 3]).reshape(-1, 1)

# 2. Input: Usiamo i valori CON SEGNO (raw), escludendo x6
# x6 è all'indice 5 nel vettore X originale (colonna 6 del CSV)
input_indices_raw = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
X_signed = X[:, input_indices_raw]

# 3. Split in Training e Validation set
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(
    X_signed, y_diff34, test_size=0.2, random_state=42
)

# 4. Scaling
scaler_x_d = StandardScaler()
X_train_d = scaler_x_d.fit_transform(X_train_d)
X_val_d = scaler_x_d.transform(X_val_d)

scaler_y_d = StandardScaler()
y_train_d = scaler_y_d.fit_transform(y_train_d)
y_val_d = scaler_y_d.transform(y_val_d)

# 5. Conversione in Tensor per PyTorch
X_train_dt = torch.FloatTensor(X_train_d)
y_train_dt = torch.FloatTensor(y_train_d)
X_val_dt = torch.FloatTensor(X_val_d)
y_val_dt = torch.FloatTensor(y_val_d)

# 6. Training (utilizzando la stessa configurazione MLP)
print("\nInizio training per y3 - y4...")
model_d, history_d, stop_epoch_d = train_model(
    config, 
    input_dim=X_signed.shape[1], 
    X_train=X_train_dt, 
    y_train=y_train_dt, 
    X_val=X_val_dt, 
    y_val=y_val_dt, 
    task_type='regression'
)

# 7. Plot e Risultati
plot_training_history(history_d, title="MLP: Predict (y3 - y4) with Signed Raw Inputs")

print(f"\nTraining terminato all'epoca: {stop_epoch_d}")
print(f"MEE finale (std) su Validation per y3 - y4: {history_d['val_score'][-1]:.6f}")

#=====================================================
# --- AGGIUNTA PER PREDIRE LA MEDIA ARITMETICA DEI VALORI ASSOLUTI DEGLI OUTPUT ---

print("\n" + "="*40)
print("INIZIO ANALISI PER: Media(|y1|, |y2|, |y3|, |y4|)")
print("="*40)

y_am_target = np.mean(np.abs(y), axis=1).reshape(-1, 1)

X_train_am, X_val_am, y_train_am, y_val_am = train_test_split(
    X_signed, y_am_target, test_size=0.2, random_state=42
)

scaler_x_am = StandardScaler()
X_train_am = scaler_x_am.fit_transform(X_train_am)
X_val_am = scaler_x_am.transform(X_val_am)

scaler_y_am = StandardScaler()
y_train_am = scaler_y_am.fit_transform(y_train_am)
y_val_am = scaler_y_am.transform(y_val_am)

X_train_amt = torch.FloatTensor(X_train_am)
y_train_amt = torch.FloatTensor(y_train_am)
X_val_amt = torch.FloatTensor(X_val_am)
y_val_amt = torch.FloatTensor(y_val_am)

print("\nInizio training per Mean(|y|)...")
model_am, history_am, stop_epoch_am = train_model(
    config, 
    input_dim=X_signed.shape[1], 
    X_train=X_train_amt, 
    y_train=y_train_amt, 
    X_val=X_val_amt, 
    y_val=y_val_amt, 
    task_type='regression'
)

# 7. Risultati e Calcolo MEE (MAE) sui dati reali
plot_training_history(history_am, title="MLP: Predict Mean(|y|) with Signed Raw Inputs")
plt.show()

print(f"\nTraining terminato all'epoca: {stop_epoch_am}")
print(f"MEE finale (std) su Validation: {history_am['val_score'][-1]:.6f}")
