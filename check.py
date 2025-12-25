import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15000  # Più epoche perché il LR è più basso per stabilità
LR = 5      # Learning rate più conservativo
PATIENCE = 2000

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data_hourglass():
    print("Loading Hourglass Data...")
    df = pd.read_csv(FILE_PATH, skiprows=7, header=None)
    data = df.iloc[:, 1:].values
    
    # Input: 11 features (No x6)
    X_all = data[:, :12]
    X_no_x6 = np.delete(X_all, 5, axis=1) # Rimuovi colonna 5 (x6)
    
    Y_raw = data[:, 12:]
    
    # Target Main: y1, y2, y4
    Y_target = Y_raw[:, [0, 1, 3]]
    
    # Target Intermedio: y3-y4
    diff_target = (Y_raw[:, 2] - Y_raw[:, 3]).reshape(-1, 1)
    
    # Split
    X_tr, X_val, Y_tr, Y_val, diff_tr, diff_val, Y_full_tr, Y_full_val = train_test_split(
        X_no_x6, Y_target, diff_target, Y_raw, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler_x = StandardScaler()
    X_tr_s = scaler_x.fit_transform(X_tr)
    X_val_s = scaler_x.transform(X_val)
    
    pt_y = PowerTransformer(method='yeo-johnson')
    Y_tr_s = pt_y.fit_transform(Y_tr)
    Y_val_s = pt_y.transform(Y_val)
    
    scaler_diff = StandardScaler()
    diff_tr_s = scaler_diff.fit_transform(diff_tr)
    diff_val_s = scaler_diff.transform(diff_val)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(Y_tr_s), torch.FloatTensor(diff_tr_s))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(Y_val_s), torch.FloatTensor(diff_val_s), torch.FloatTensor(Y_full_val))
    
    return train_ds, val_ds, pt_y, scaler_diff

# ==========================================
# 3. MODELLO CLESSIDRA (STABILE)
# ==========================================
class HourglassNet(nn.Module):
    def __init__(self, input_dim=11, output_dim=3):
        super(HourglassNet, self).__init__()
        
        # ENCODER: Comprime tutto in 1 singola variabile latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), # Stabilizza il training
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Linear(32, 1) # IL COLLO DI BOTTIGLIA (y3-y4)
        )
        
        # DECODER: Espande la variabile latente verso l'output
        # Sostituisce il layer polinomiale con una rete neurale profonda
        # che impara la funzione geometrica in modo stabile.
        self.decoder = nn.Sequential(
            nn.Linear(1, 128), # Espansione brutale: 1 -> 128
            nn.GELU(),         # Non-linearità
            
            nn.Linear(128, 256),
            nn.GELU(),
            
            nn.Linear(256, 128),
            nn.GELU(),
            
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        latent_diff = self.encoder(x)
        out = self.decoder(latent_diff)
        return out, latent_diff

# ==========================================
# 4. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    train_ds, val_ds, pt_y, scaler_diff = load_data_hourglass()
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HourglassNet(input_dim=11).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    
    # Loss
    criterion_main = nn.HuberLoss(delta=1.0)
    criterion_aux = nn.MSELoss()
    
    best_mee = float('inf')
    no_imp = 0
    
    print("\n--- Start Hourglass Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for bx, by, b_diff in train_dl:
            bx, by, b_diff = bx.to(DEVICE), by.to(DEVICE), b_diff.to(DEVICE)
            
            optimizer.zero_grad()
            out_pred, latent_pred = model(bx)
            
            # 1. Loss Principale (Output)
            loss_y = criterion_main(out_pred, by)
            
            # 2. Loss Ausiliaria (Fisica)
            # Forza il neurone centrale a comportarsi come y3-y4
            loss_diff = 0#criterion_aux(latent_pred, b_diff)
            
            # Somma pesata
            loss = loss_y + (0.5 * loss_diff)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        mee_sum = 0
        diff_error_sum = 0
        count = 0
        
        with torch.no_grad():
            for bx, _, b_diff, by_full_real in val_dl:
                bx, b_diff = bx.to(DEVICE), b_diff.to(DEVICE)
                
                out_pred_s, latent_pred_s = model(bx)
                
                # Monitoriamo quanto bene impara la fisica
                diff_error_sum += criterion_aux(latent_pred_s, b_diff).item() * bx.size(0)
                
                # MEE Reale
                out_np = out_pred_s.cpu().numpy()
                out_real = pt_y.inverse_transform(out_np)
                y1, y2, y4 = out_real[:, 0], out_real[:, 1], out_real[:, 2]
                
                # Ricostruzione y3 (usando la diff PREDETTA INTERNAMENTE)
                # Dobbiamo de-scalarla prima
                diff_pred_np = latent_pred_s.cpu().numpy()
                diff_pred_real = scaler_diff.inverse_transform(diff_pred_np).flatten()
                
                y3 = y4 + diff_pred_real
                
                pred_full = np.column_stack((y1, y2, y4))
                err = np.linalg.norm(pred_full[:,[0,1,2]] - by_full_real.numpy()[:,[0,1,3]], axis=1).mean()
                mee_sum += err * bx.size(0)
                count += bx.size(0)
                
        val_mee = mee_sum / count
        print("mee: ", val_mee)
        val_diff_mse = diff_error_sum / count
        
        scheduler.step(val_mee)
        
        if val_mee < best_mee:
            best_mee = val_mee
            no_imp = 0
            torch.save(model.state_dict(), "best_hourglass.pth")
            if (epoch+1) % 50 == 0:
                print(f"Ep {epoch+1:04d} | MEE: {best_mee:.4f} | Diff MSE: {val_diff_mse:.4f}")
        else:
            no_imp += 1
            
        if no_imp >= PATIENCE:
            print(f"Early Stopping. Best MEE: {best_mee:.4f}")
            break
            
    print(f"Risultato Finale MEE: {best_mee:.4f}")
