import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import load_cup_data
# Importiamo train_model dal file mlp.py aggiornato
from mlp import train_model 

# Configurazione Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # ==========================================
    # 1. CARICAMENTO E PREPROCESSING
    # ==========================================
    # Carichiamo il Training Set fornito
    X_cup, y_cup = load_cup_data('data/CUP/ML-CUP25-TR.csv')

    # Creiamo uno split interno (Train/Test) per valutare le prestazioni
    # Usiamo il 20% come test set interno
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_cup, y_cup, test_size=0.2, random_state=42
    )

    # Scaling delle Feature (X)
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_test = scaler_x.transform(X_test_raw)

    # Scaling del Target (y)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    # Scaliamo anche y_test per il calcolo della loss interna (val_loss)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # Conversione in Tensori
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    y_test_scaled_t = torch.FloatTensor(y_test_scaled).to(device)

    # ==========================================
    # 2. CONFIGURAZIONE MODELLO (OTTIMIZZATA)
    # ==========================================
    # Parametri aggressivi per scendere sotto MEE 1.0
    # Usiamo Dropout e Scheduler per gestire reti grandi e deep.
    configs = [
            {'hidden_layers': [128, 512, 256, 128], 
             'activation': 'LeakyReLU', 
             'lr': 0.005, 
             'weight_decay': 0.0001, 
             'dropout': 0.0, 
             'batch_size': 64, 
             'epochs': 800, 
             'es': True, 
             'patience': 50, 
             'loss': 'MSE', 
             'use_scheduler': True, 
             'momentum': 0.9},
             {'hidden_layers': [128, 256, 128], 
              'activation': 'LeakyReLU', 
              'lr': 0.005, 
              'weight_decay': 0.0001, 
              'dropout': 0.1, 
              'batch_size': 64, 
              'epochs': 800, 
              'es': True, 
              'patience': 50, 
              'loss': 'MSE', 
              'use_scheduler': True, 
              'momentum': 0.9},
             {'hidden_layers': [128, 256, 128], 
              'activation': 'LeakyReLU', 
              'lr': 0.001, 
              'weight_decay': 0.0001, 
              'dropout': 0.0, 
              'batch_size': 64, 
              'epochs': 800, 
              'es': True, 
              'patience': 50, 
              'loss': 'MSE', 
              'use_scheduler': True, 
              'momentum': 0.9},
             {'hidden_layers': [128, 512, 256, 128], 
              'activation': 'LeakyReLU', 
              'lr': 0.001, 
              'weight_decay': 0.0001, 
              'dropout': 0.0, 
              'batch_size': 64, 
              'epochs': 800, 
              'es': True, 
              'patience': 50, 'loss': 'MSE', 
              'use_scheduler': True, 
              'momentum': 0.9}
            ]
    NUM_ENSEMBLE = 5  # Numero di modelli da addestrare per OGNI target (Totale 20 modelli)
    
    # Matrice per raccogliere le predizioni finali (N_samples, 4_targets)
    final_preds_scaled = np.zeros((X_test.shape[0], 4))

    print(f"\nInizio Training Ensemble: {NUM_ENSEMBLE} modelli x 4 Target = {NUM_ENSEMBLE*4} Training totali.")
    print("-" * 60)

    # ==========================================
    # 3. TRAINING LOOP (PER TARGET)
    # ==========================================
    for i in range(4):
        print(f"🎯 TARGET {i+1} / 4")
        
        # Selezione colonna target singolo
        y_train_single = y_train_t[:, i:i+1]
        y_test_single = y_test_scaled_t[:, i:i+1]
        
        # Configurazione specifica per questo target
        config = configs[i]
        
        # Accumulatore per le predizioni di questo target (per fare la media)
        target_preds_sum = np.zeros((X_test.shape[0], 1))
        
        # Training dei modelli dell'ensemble
        for k in range(NUM_ENSEMBLE):
            print(f"   ∟ Modello {k+1}/{NUM_ENSEMBLE} ... ", end="")
            
            # Addestramento
            model, hist, stop_epoch = train_model(
                config, 
                X_train.shape[1], 
                X_train_t, 
                y_train_single, 
                X_test_t,       # Usiamo X_test come Validation per monitorare
                y_test_single, 
                task_type='regression'
            )
            
            # Predizione sul Test Set
            model.eval()
            with torch.no_grad():
                # Spostiamo su CPU per numpy
                pred_t = model(X_test_t)
                pred_np = pred_t.cpu().numpy()
            
            # Aggiungi alla somma
            target_preds_sum += pred_np
            
            # Feedback rapido
            best_val_loss = np.min(hist['val_loss'])
            print(f"Done. (Stop Ep: {stop_epoch}, Best Loss: {best_val_loss:.5f})")
            
        # Media delle predizioni per il target i
        avg_preds = target_preds_sum / NUM_ENSEMBLE
        final_preds_scaled[:, i:i+1] = avg_preds
        print(f"✅ Target {i+1} completato.\n")

   # ==========================================
    # 4. VALUTAZIONE SU DATI STANDARDIZZATI
    # ==========================================
    print("=" * 60)
    print("RISULTATI FINALI (SCALA STANDARDIZZATA)")
    print("=" * 60)

    # Calcolo errore diretto tra predizioni scalate e ground truth scalato
    errors_scaled = final_preds_scaled - y_test_scaled
    
    # MEE Scalato
    norms_scaled = np.linalg.norm(errors_scaled, axis=1)
    mee_score_scaled = np.mean(norms_scaled)
    
    # MSE Scalato (utile per confronto con la Loss)
    mse_score_scaled = np.mean(errors_scaled**2)

    print(f"MEE : {mee_score_scaled:.6f}")
    print(f"MSE : {mse_score_scaled:.6f}")
    
    print("-" * 30)
    print("Dettaglio errore medio (MAE) per componente:")
    mae_per_col = np.mean(np.abs(errors_scaled), axis=0)
    for idx, mae in enumerate(mae_per_col):
        print(f"Target {idx+1}: {mae:.6f}")

if __name__ == "__main__":
    main()
