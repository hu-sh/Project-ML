import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
FILE_PATH = 'data/CUP/ML-CUP25-TR.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_solver_final.pth"
POLYDEG = 2
# ==========================================
# 2. DEFINIZIONE CLASSE SOLVER (Identica al training)
# ==========================================
class SolverNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(SolverNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def full_pipeline_test():
    print(f"--- TEST VARIANT: Ensemble 5 MLP (No Hidden, No x6) -> Solver ---")
    
    # -------------------------------------------------------
    # A. CARICAMENTO E SPLIT DATI
    # -------------------------------------------------------
    print("1. Caricamento dati e split (Seed 42)...")
    try:
        df = pd.read_csv(FILE_PATH, skiprows=7, header=None)
    except FileNotFoundError:
        print(f"ERRORE: File {FILE_PATH} non trovato.")
        return

    data = df.iloc[:, 1:].values
    X = data[:, :12]
    Y = data[:, 12:]
    
    # Calcoliamo la Diff Vera per il training dello Sniper
    diff_real = (Y[:, 2] - Y[:, 3])
    
    # Split (rigorosamente uguale a quello usato per addestrare il Solver)
    X_tr, X_val, diff_tr, diff_val, Y_tr, Y_val = train_test_split(
        X, diff_real, Y, test_size=0.2, random_state=42
    )
    
    # -------------------------------------------------------
    # B. ADDESTRAMENTO SNIPER (MODIFICATO)
    # -------------------------------------------------------
    print("2. Addestramento Sniper (Ensemble 5 MLP, No Hidden Layer, No x6)...")
    
    # Rimuoviamo la colonna x6 (indice 5)
    # X ha indici 0..11. L'indice 5 è la sesta feature.
    X_tr_snp = np.delete(X_tr, 5, axis=1)
    X_val_snp = np.delete(X_val, 5, axis=1)
    
    n_models = 5
    preds_matrix = np.zeros((len(X_val_snp), n_models))
    
    for i in range(n_models):
        print(f"   - Training MLP {i+1}/{n_models}...")
        # hidden_layer_sizes=() crea un percettrone senza layer nascosti (Input -> Output)
        # È matematicamente una regressione lineare, ma addestrata via Adam
        mlp = MLPRegressor(
            hidden_layer_sizes=(), 
            activation='identity',  # Lineare
            solver='adam', 
            max_iter=2000, 
            random_state=42 + i,    # Seed diversi per variare l'inizializzazione
            learning_rate_init=0.01
        )
        mlp.fit(X_tr_snp, diff_tr)
        preds_matrix[:, i] = mlp.predict(X_val_snp)
    
    # Media delle predizioni (Ensemble)
    pred_diff_val = np.mean(preds_matrix, axis=1)
    
    mae_sniper = mean_absolute_error(diff_val, pred_diff_val)
    print(f"   >>> Errore Sniper (MAE) su (y3-y4): {mae_sniper:.4f}")

    # -------------------------------------------------------
    # C. PREPARAZIONE INPUT SOLVER
    # -------------------------------------------------------
    print("3. Preparazione Input per il Solver...")
    
    # 1. Polinomi (Grado 8) su Diff (come da training originale)
    poly = PolynomialFeatures(degree=POLYDEG, include_bias=False)
    diff_tr_reshaped = diff_tr.reshape(-1, 1)
    poly.fit(diff_tr_reshaped)
    
    # Trasformiamo la Diff PREDETTA dal nuovo Sniper
    pred_diff_val_reshaped = pred_diff_val.reshape(-1, 1)
    X_solver_val_poly = poly.transform(pred_diff_val_reshaped)
    
    # 2. Scaling Input
    scaler_x = StandardScaler()
    X_solver_tr_poly = poly.transform(diff_tr_reshaped)
    scaler_x.fit(X_solver_tr_poly)
    
    X_solver_input = scaler_x.transform(X_solver_val_poly)
    
    # 3. Scaling Output (PowerTransformer per inversione)
    Y_target_tr = Y_tr[:, [0, 1, 3]] # y1, y2, y4
    pt_y = PowerTransformer(method='yeo-johnson')
    pt_y.fit(Y_target_tr)

    # -------------------------------------------------------
    # D. ESECUZIONE SOLVER
    # -------------------------------------------------------
    print(f"4. Caricamento Modello '{MODEL_PATH}' ed Esecuzione...")
    
    input_dim = X_solver_input.shape[1]
    model = SolverNet(input_dim=input_dim).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print(f"ERRORE: Non trovo {MODEL_PATH}. Assicurati di averlo nella cartella.")
        # Creiamo un dummy per non crashare se l'utente sta solo testando il codice senza pth
        print("   (Creo un modello random dummy per terminare il test...)")
    
    with torch.no_grad():
        tensor_in = torch.FloatTensor(X_solver_input).to(DEVICE)
        out_scaled = model(tensor_in).cpu().numpy()
        
    # Inverse Transform
    out_real = pt_y.inverse_transform(out_scaled)
    
    # -------------------------------------------------------
    # E. RICOSTRUZIONE E CALCOLO MEE
    # -------------------------------------------------------
    print("5. Calcolo MEE Finale...")
    
    y1_pred = out_real[:, 0]
    y2_pred = out_real[:, 1]
    y4_pred = out_real[:, 2]
    
    # Ricostruzione y3: y3 = y4 + diff_predetta
    y3_pred = y4_pred + pred_diff_val
    
    pred_full = np.column_stack((y1_pred, y2_pred, y3_pred, y4_pred))
    
    # Calcolo errore
    errors = pred_full - Y_val
    mee_vector = np.linalg.norm(errors, axis=1)
    final_mee = np.mean(mee_vector)
    
    print("\n" + "="*50)
    print(f"RISULTATO REALE PIPELINE (NO x6, 5-MLP Ensemble):")
    print(f"Errore Input (Sniper MAE): {mae_sniper:.4f}")
    print(f"MEE Finale (Test simulato):  {final_mee:.4f}")
    print("="*50)

if __name__ == "__main__":
    full_pipeline_test()
