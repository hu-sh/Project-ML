import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer

# ==========================================
# CONFIGURAZIONE
# ==========================================
FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 2000  # Tante epoche per limare il minimo assoluto
LR = 1e-3
PATIENCE = 200

# ==========================================
# DATA LOADING (SOLO y3-y4)
# ==========================================
def load_data_solver():
    print("Loading Solver Data: Input = (y3-y4)...")
    df = pd.read_csv(FILE_PATH, skiprows=7, header=None)
    data = df.iloc[:, 1:].values
    Y_raw = data[:, 12:] 
    
    # INPUT: La tua diff
    diff_input = (Y_raw[:, 2] - Y_raw[:, 3]).reshape(-1, 1)
    
    # FEATURE ENGINEERING
    POLY_DEGREE = 2 
    print(f"  -> Espansione Polinomiale Grado {POLY_DEGREE}...")
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(diff_input)
    
    # TARGET
    Y_target = Y_raw[:, [0, 1, 3]]
    
    # SPLIT: 4 input arrays -> 8 output arrays
    # Ho aggiunto Y_full_tr e diff_tr che mancavano
    X_tr, X_val, Y_tr, Y_val, Y_full_tr, Y_full_val, diff_tr, diff_val = train_test_split(
        X_poly, Y_target, Y_raw, diff_input, test_size=0.2, random_state=42
    )
    
    # SCALING
    scaler_x = StandardScaler()
    X_tr_s = scaler_x.fit_transform(X_tr)
    X_val_s = scaler_x.transform(X_val)
    
    pt_y = PowerTransformer(method='yeo-johnson')
    Y_tr_s = pt_y.fit_transform(Y_tr)
    Y_val_s = pt_y.transform(Y_val)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(Y_tr_s))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(Y_val_s), torch.FloatTensor(Y_full_val), torch.FloatTensor(diff_val))
    
    return train_ds, val_ds, X_poly.shape[1], pt_y, scaler_x, poly 
# ==========================================
# MODELLO
# ==========================================
class SolverNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(SolverNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(), # GELU è più liscia di ReLU, meglio per approssimare curve fisiche
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# METRICA MEE
# ==========================================
def get_mee(y_pred_s, y_full_real, diff_val, pt):
    # De-transform Yeo-Johnson
    y_pred_np = y_pred_s.detach().cpu().numpy()
    y_pred_r = pt.inverse_transform(y_pred_np)
    y_pred_r = torch.FloatTensor(y_pred_r).to(DEVICE)
    
    y1, y2, y4 = y_pred_r[:, 0], y_pred_r[:, 1], y_pred_r[:, 2]
    
    # Ricostruzione y3 = y4 + diff_input
    y3 = y4 + diff_val.squeeze()
    
    pred_full = torch.stack([y1, y2, y3, y4], dim=1)
    return torch.norm(pred_full - y_full_real, p=2, dim=1).mean().item()

# ==========================================
# TRAINING
# ==========================================
if __name__ == "__main__":
    train_ds, val_ds, input_dim, pt_y, scaler_x, poly = load_data_solver()
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SolverNet(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    
    # Huber Loss con delta basso per raffinare la precisione
    criterion = nn.HuberLoss(delta=0.5) 
    
    best_mee = float('inf')
    no_imp = 0
    
    print(f"\n--- Start Training Solver (Target MEE < 2.5) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        mee_sum = 0
        with torch.no_grad():
            for bx, _, by_full, b_diff in val_dl:
                bx, by_full, b_diff = bx.to(DEVICE), by_full.to(DEVICE), b_diff.to(DEVICE)
                out = model(bx)
                mee_sum += get_mee(out, by_full, b_diff, pt_y) * bx.size(0)
        
        val_mee = mee_sum / len(val_ds)
        scheduler.step(val_mee)
        
        if val_mee < best_mee:
            best_mee = val_mee
            no_imp = 0
            torch.save(model.state_dict(), "best_solver_final.pth")
            if (epoch+1) % 50 == 0:
                print(f"Ep {epoch+1:04d} | NEW BEST MEE: {best_mee:.4f}")
        else:
            no_imp += 1
            
        if no_imp >= PATIENCE:
            print(f"Early Stopping. Best MEE: {best_mee:.4f}")
            break
            
    print(f"\nRISULTATO FINALE SOLVER: {best_mee:.4f}")
    
    # --- SALVATAGGIO OGGETTI PER INFERENZA ---
    # Per usare questo modello sul test set (con la tua predizione di y3-y4), 
    # ti serviranno lo scaler e il trasformatore polinomiale.
    import joblib
    joblib.dump(scaler_x, 'scaler_x_solver.pkl')
    joblib.dump(pt_y, 'pt_y_solver.pkl')
    joblib.dump(poly, 'poly_solver.pkl')
    print("Modello e trasformatori salvati. Usa 'best_solver_final.pth' con la tua stima di y3-y4.")
