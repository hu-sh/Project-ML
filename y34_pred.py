import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. MODELLO RESIDUALE AVANZATO
# ==============================================================================
class DeepResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(DeepResidualMLP, self).__init__()
        
        # Ramo Lineare (Skip) - Il tuo "0.99" di base
        self.skip = nn.Linear(input_size, 1)
        
        # Ramo Non-Lineare più profondo
        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layer = nn.Linear(in_dim, h_dim)
            # Kaiming init per ReLU/LeakyReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.BatchNorm1d(h_dim)) # Stabilizza il training
            in_dim = h_dim
        
        # Output del ramo residuo inizializzato quasi a zero
        self.res_out = nn.Linear(in_dim, 1)
        nn.init.constant_(self.res_out.weight, 0)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.skip(x) + self.res_out(self.mlp(x))

# ==============================================================================
# 2. CARICAMENTO E FEATURE ENGINEERING (Aggiunta Termini Quadratici)
# ==============================================================================
def load_data(path):
    data = pd.read_csv(path, header=None, comment='#').values
    X = data[:, 1:13]
    # Target y3 - y4
    y = (data[:, 15] - data[:, 16]).reshape(-1, 1)
    
    # Aggiungiamo termini quadratici per le feature più importanti (es. x1..x12)
    # Questo permette al modello di catturare curvature senza hidden layers enormi
    X_quad = np.hstack([X, X**2]) 
    return X_quad, y

PATH = 'data/CUP/ML-CUP25-TR.csv'
X_all, y_all = load_data(PATH)

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_std = scaler.fit_transform(X_train)
X_vl_std = scaler.transform(X_val)

X_tr_t = torch.FloatTensor(X_tr_std).to(device)
y_tr_t = torch.FloatTensor(y_train).to(device)
X_vl_t = torch.FloatTensor(X_vl_std).to(device)
y_vl_t = torch.FloatTensor(y_val).to(device)

# ==============================================================================
# 3. TRAINING CON HUBER LOSS E SCHEDULER
# ==============================================================================
model = DeepResidualMLP(X_tr_std.shape[1], [64, 32, 16]).to(device)

# HuberLoss: si comporta come MSE vicino allo zero e come MAE lontano
# Aiuta a ignorare gli outlier che tengono alto il MEE
criterion = nn.HuberLoss(delta=1.0) 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

best_val_mee = float('inf')

for epoch in range(2000):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        v_preds = model(X_vl_t)
        # MEE (Mean Euclidean Error)
        val_mee = torch.abs(v_preds - y_vl_t).mean().item()
        
    scheduler.step(val_mee)
    
    if val_mee < best_val_mee:
        best_val_mee = val_mee
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Val MEE: {val_mee:.5f} | Best: {best_val_mee:.5f} | LR: {optimizer.param_groups[0]['lr']}")

print(f"\nTarget Raggiunto? Best Val MEE: {best_val_mee:.5f}")
