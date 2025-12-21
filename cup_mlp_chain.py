import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from utils import load_cup_data
# Assicurati che mlp.py sia aggiornato con le classi ResidualBlock o DynamicMLP
from mlp import train_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_oof_predictions(config, X, y, k_folds=5):
    """
    Genera predizioni Out-Of-Fold (OOF) per evitare Data Leakage nella catena.
    Divide il training set in k fold. Per ogni fold, addestra un modello sugli altri k-1
    e predice sul fold corrente.
    Restituisce un vettore di predizioni 'pulite' della stessa lunghezza di X.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((X.shape[0], 1))
    
    # Convertiamo tutto in tensori una volta sola per efficienza
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).to(device)
    
    # Input dimension
    input_dim = X.shape[1]
    
    print(f"      [OOF] Generating {k_folds}-fold preds...", end="")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split dati
        X_train_fold = X_t[train_idx]
        y_train_fold = y_t[train_idx]
        X_val_fold = X_t[val_idx]
        y_val_fold = y_t[val_idx]
        
        # Training modello temporaneo
        # Nota: usiamo X_val_fold anche come validation per l'early stopping
        model, _, _ = train_model(
            config, 
            input_dim, 
            X_train_fold, 
            y_train_fold, 
            X_val_fold, 
            y_val_fold, 
            task_type='regression'
        )
        
        # Predizione sul fold di validazione (che il modello non ha visto in training)
        model.eval()
        with torch.no_grad():
            pred = model(X_val_fold).cpu().numpy()
            
        oof_preds[val_idx] = pred
        print(".", end="")
        
    print(" Done.")
    return oof_preds

def main():
    # ==========================================
    # 1. CARICAMENTO E PREPROCESSING
    # ==========================================
    X_cup, y_cup = load_cup_data('data/CUP/ML-CUP25-TR.csv')
    
    # Split 80/20 standard
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_cup, y_cup, test_size=0.2, random_state=42
    )

    # Scaling
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_test = scaler_x.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # ==========================================
    # 2. CONFIGURAZIONI (OTTIMIZZATE CON ADAM)
    # ==========================================
    # Se non le hai ancora, usa queste di default robuste.
    
    configs = {
    1: {'hidden_layers': [512, 256, 128], 'activation': 'LeakyReLU', 'lr': 0.005, 'dropout': 0.1, 'loss': 'MSE', 'weight_decay': 0.0001, 'epochs': 600, 'batch_size': 64, 'es': True, 'patience': 40, 'use_scheduler': True},
    0: {'hidden_layers': [512, 256, 128], 'activation': 'LeakyReLU', 'lr': 0.005, 'dropout': 0.1, 'loss': 'MSE', 'weight_decay': 0.0001, 'epochs': 600, 'batch_size': 64, 'es': True, 'patience': 40, 'use_scheduler': True},
    3: {'hidden_layers': [512, 256, 128], 'activation': 'LeakyReLU', 'lr': 0.005, 'dropout': 0.2, 'loss': 'MSE', 'weight_decay': 0.0001, 'epochs': 600, 'batch_size': 64, 'es': True, 'patience': 40, 'use_scheduler': True},
    2: {'hidden_layers': [512, 256, 128], 'activation': 'LeakyReLU', 'lr': 0.001, 'dropout': 0.1, 'loss': 'MSE', 'weight_decay': 0.0001, 'epochs': 600, 'batch_size': 64, 'es': True, 'patience': 40, 'use_scheduler': True},
}

    chain_order = [1, 0, 3, 2]

    
    # Numero di modelli per l'Ensemble Finale di ogni step
    # Aumentalo a 10 o 20 per la sottomissione finale
    NUM_ENSEMBLE = 5
    
    # ==========================================
    # 3. CHAIN LOOP (CON OOF)
    # ==========================================
    # Accumulatori features (iniziano con le feature originali numpy)
    X_train_curr = X_train.copy()
    X_test_curr = X_test.copy()

    # Dizionario per salvare le predizioni finali {target_idx: prediction_vector}
    chain_test_predictions = {}

    print(f"\nInizio Training CHAIN con OOF ({NUM_ENSEMBLE} modelli finali x 4 step)...")
    print(f"Ordine Catena: Target {np.array(chain_order)+1}")
    print("-" * 60)

    for step, target_idx in enumerate(chain_order):
        print(f"🔗 STEP {step+1}: Training Target {target_idx+1}")
        
        # Dati target correnti
        y_tr_single = y_train[:, target_idx:target_idx+1] # Numpy
        y_te_single = y_test_scaled[:, target_idx:target_idx+1] # Numpy
        
        # Config per questo step
        config = configs[target_idx]

        # --- FASE A: GENERAZIONE FEATURE PER IL TRAINING (OOF) ---
        # Questa è la parte critica: generiamo le predizioni sul training set
        # usando modelli che NON hanno visto quei dati.
        print(f"   ⚙️ Generazione OOF features (per nutrire il prossimo step)...")
        # Nota: Qui usiamo 1 sola run di OOF (5 fold) per generare le feature.
        # È sufficiente. Non serve fare ensemble di OOF (sarebbe lentissimo).
        oof_train_pred = get_oof_predictions(config, X_train_curr, y_tr_single, k_folds=5)
        
        # --- FASE B: TRAINING ENSEMBLE PER IL TEST SET ---
        # Ora addestriamo i modelli "veri" su TUTTO il training set corrente
        # per predire il test set.
        print(f"   🚀 Training Ensemble Finale ({NUM_ENSEMBLE} modelli)...", end="")
        
        test_preds_sum = np.zeros((X_test.shape[0], 1))
        
        # Convertiamo dataset corrente in tensori per il training loop
        X_tr_t = torch.FloatTensor(X_train_curr).to(device)
        y_tr_t = torch.FloatTensor(y_tr_single).to(device)
        X_te_t = torch.FloatTensor(X_test_curr).to(device)
        y_te_t = torch.FloatTensor(y_te_single).to(device) # Solo per validation score
        
        current_input_dim = X_train_curr.shape[1]

        best_losses = []
        for k in range(NUM_ENSEMBLE):
            model, hist, _ = train_model(
                config, 
                current_input_dim, 
                X_tr_t, 
                y_tr_t, 
                X_te_t, 
                y_te_t, 
                task_type='regression'
            )
            
            model.eval()
            with torch.no_grad():
                pred = model(X_te_t).cpu().numpy()
            
            test_preds_sum += pred
            best_losses.append(np.min(hist['val_loss']))
        
        print(f" Done. (Avg Best Loss: {np.mean(best_losses):.5f})")
        
        # Media Ensemble Test
        avg_test_pred = test_preds_sum / NUM_ENSEMBLE
        
        # Salviamo la predizione finale per questo target
        chain_test_predictions[target_idx] = avg_test_pred

        X_train_curr = np.hstack([X_train_curr, oof_train_pred])
        X_test_curr = np.hstack([X_test_curr, avg_test_pred])
        
        print(f"   ✅ Feature aggiunte. Nuovo input size: {X_train_curr.shape[1]}\n")

    # ==========================================
    # 4. VALUTAZIONE FINALE
    # ==========================================
    print("=" * 60)
    print("RISULTATI FINALI (CHAIN + OOF + ADAM)")
    print("=" * 60)
    
    # Ricostruzione matrice predizioni (N, 4) ordinata corretta
    final_preds_scaled = np.zeros((X_test.shape[0], 4))
    for i in range(4):
        final_preds_scaled[:, i:i+1] = chain_test_predictions[i]

    # Errori Scalati
    errors_scaled = final_preds_scaled - y_test_scaled
    norms_scaled = np.linalg.norm(errors_scaled, axis=1)
    mee_scaled = np.mean(norms_scaled)
    mse_scaled = np.mean(errors_scaled**2)

    print(f"MEE (Standardized): {mee_scaled:.6f}")
    print(f"MSE (Standardized): {mse_scaled:.6f}")
    print("-" * 30)
    
    # MAE per componente
    mae_per_col = np.mean(np.abs(errors_scaled), axis=0)
    for idx, mae in enumerate(mae_per_col):
        print(f"Target {idx+1} MAE: {mae:.6f}")

if __name__ == "__main__":
    main()
