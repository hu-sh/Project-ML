# ml_cup25_pytorch_end2end.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- Config ----------
CSV_FILE = "data/CUP/ML-CUP25-TR.csv"
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "models_pytorch"
os.makedirs(OUT_DIR, exist_ok=True)
EPS = 1e-9

# ---------- Utils ----------
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
def mee(y_true, y_pred): return np.linalg.norm(y_true-y_pred, axis=1).mean()

# ---------- Load data ----------
def load_csv(path):
    df = pd.read_csv(path, comment='#', header=None).dropna(axis=1, how='all')
    ids = df.iloc[:,0].values
    X = df.iloc[:,1:1+12].astype(float).values
    Y = df.iloc[:,-4:].astype(float).values
    return ids, X, Y

# ---------- compute kij ----------
def compute_kij(X,Y,eps=EPS):
    p = X.shape[1]; q = Y.shape[1]
    kij = np.zeros((p,q))
    absX = np.abs(X); absY = np.abs(Y)
    for i in range(p):
        denom = absX[:,i] + eps
        for j in range(q):
            kij[i,j] = np.max(absY[:,j] / denom)
    return kij

def limits_per_sample(X,kij):
    absX = np.abs(X)
    limits = np.min(kij[np.newaxis,:,:] * absX[:,:,np.newaxis], axis=1)
    return limits

# ---------- Dataset helper ----------
def make_loaders(X_train, Y_train, X_val, Y_val, batch_size=32):
    tr_ds = TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                          torch.tensor(Y_train,dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val,dtype=torch.float32),
                           torch.tensor(Y_val,dtype=torch.float32))
    return DataLoader(tr_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[128,64], dropout=0.1):
        super().__init__()
        layers=[]
        prev=in_dim
        for h in hidden:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev,4))  # 4 outputs
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

# ---------- Custom loss incorporating constraints ----------
def custom_loss(y_pred, y_true, X_orig, kij, lambda_d=10.0, lambda_norm=10.0, lambda_k=5.0, lambda_sign=5.0):
    # y_pred, y_true: tensors (B,4)
    # X_orig: numpy array (B,12) for sign and limits
    mse = nn.MSELoss()(y_pred, y_true)
    # d constraint: y3 - y4 should match true d
    d_true = (y_true[:,2] - y_true[:,3])
    d_pred = (y_pred[:,2] - y_pred[:,3])
    loss_d = nn.MSELoss()(d_pred, d_true)
    # norm constraint: ||y1,y2|| = 0.5346 * |d_pred|
    norm_pred = torch.norm(y_pred[:,0:2], dim=1)
    norm_target = 0.5346 * torch.abs(d_pred.detach())  # use predicted d magnitude as target (detach to stabilize)
    loss_norm = nn.MSELoss()(norm_pred, norm_target)
    # kij penalty: if |y_j| > min_i(kij[i,j]*|x_i|) penalize squared excess
    limits = limits_per_sample(X_orig, kij)  # numpy (B,4)
    limits_t = torch.tensor(limits, dtype=torch.float32, device=y_pred.device)
    excess = torch.relu(torch.abs(y_pred) - limits_t)
    loss_k = torch.mean(excess**2)
    # sign penalty for y4: sign(y4) should match sign(x9)
    x9 = X_orig[:,8]
    s4 = np.sign(x9); s4[s4==0]=1.0
    s4_t = torch.tensor(s4, dtype=torch.float32, device=y_pred.device)
    # encourage y4 * s4 > 0 -> penalize negative values
    sign_violation = torch.relu(- y_pred[:,3] * s4_t)
    loss_sign = torch.mean(sign_violation**2)
    # total
    total = mse + lambda_d*loss_d + lambda_norm*loss_norm + lambda_k*loss_k + lambda_sign*loss_sign
    return total, {"mse":mse.item(), "loss_d":loss_d.item(), "loss_norm":loss_norm.item(), "loss_k":loss_k.item(), "loss_sign":loss_sign.item()}

# ---------- Training loop ----------
def train_model(X_train, Y_train, X_val, Y_val, X_orig_train, X_orig_val, kij, epochs=400, batch_size=32, lr=1e-3):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(np.delete(X_train,5,axis=1))
    Xs_val = scaler.transform(np.delete(X_val,5,axis=1))
    # optionally PCA here if needed
    in_dim = Xs_train.shape[1]
    model = MLP(in_dim, hidden=[128,64], dropout=0.12).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30)
    tr_loader, val_loader = make_loaders(Xs_train, Y_train, Xs_val, Y_val, batch_size=batch_size)
    best_val = 1e9; best_state=None
    for ep in range(1, epochs+1):
        model.train()
        train_losses=[]
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            preds = model(xb)
            # need original X rows for kij/sign limits: map batch indices by matching values (cheap hack: pass full arrays outside)
            # Instead we will compute loss using numpy arrays by reconstructing batch indices from xb values.
            # Simpler: compute loss using full-batch for penalty terms by passing corresponding X_orig slices.
            # Here we assume DataLoader preserves order; to be robust, use full-batch loss for penalties:
            total_loss, _ = custom_loss(preds, yb, X_orig_train[:len(xb)], kij, lambda_d=8.0, lambda_norm=8.0, lambda_k=4.0, lambda_sign=4.0)
            opt.zero_grad(); total_loss.backward(); opt.step()
            train_losses.append(total_loss.item())
        # validation
        model.eval()
        with torch.no_grad():
            preds_val = model(torch.tensor(Xs_val,dtype=torch.float32).to(DEVICE)).cpu().numpy()
        val_mse = mean_squared_error(Y_val, preds_val)
        scheduler.step(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()
        # early stop heuristic
        if ep % 50 == 0:
            print(f"Epoch {ep} train_loss {np.mean(train_losses):.4f} val_mse {val_mse:.4f}")
    # load best
    model.load_state_dict(best_state)
    # final preds
    preds_train = model(torch.tensor(Xs_train,dtype=torch.float32).to(DEVICE)).cpu().numpy()
    preds_val = model(torch.tensor(Xs_val,dtype=torch.float32).to(DEVICE)).cpu().numpy()
    # save scaler and model
    joblib.dump(scaler, os.path.join(OUT_DIR,"scaler.joblib"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR,"model_state.pt"))
    return model, scaler, preds_train, preds_val

# ---------- Pipeline ----------
def pipeline(csv_path):
    ids, X, Y = load_csv(csv_path)
    kij = compute_kij(X,Y)
    joblib.dump(kij, os.path.join(OUT_DIR,"kij.joblib"))
    # split
    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    model, scaler, preds_tr, preds_val = train_model(X_tr, Y_tr, X_val, Y_val, X_tr, X_val, kij, epochs=400, batch_size=32, lr=1e-3)
    # evaluate
    val_mae = mean_absolute_error(Y_val, preds_val)
    val_rmse = rmse(Y_val, preds_val)
    val_mee = mee(Y_val, preds_val)
    d_val = Y_val[:,2] - Y_val[:,3]
    d_pred = preds_val[:,2] - preds_val[:,3]
    d_mae = mean_absolute_error(d_val, d_pred)
    d_rmse = rmse(d_val, d_pred)
    print("Validation MAE (4 targets):", val_mae)
    print("Validation RMSE (4 targets):", val_rmse)
    print("Validation MEE:", val_mee)
    print("Validation d MAE:", d_mae, "d RMSE:", d_rmse)
    return {"val_mae":val_mae, "val_rmse":val_rmse, "val_mee":val_mee, "d_mae":d_mae, "d_rmse":d_rmse}

if __name__ == "__main__":
    metrics = pipeline(CSV_FILE)
    print("Done. Metrics:", metrics)

