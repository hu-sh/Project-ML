import torch
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import load_cup_data
from mlp import train_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("--- RICERCA ORDINE OTTIMALE CHAIN ---")
    
    # 1. SETUP DATI (Identico)
    X_cup, y_cup = load_cup_data('data/CUP/ML-CUP25-TR.csv')
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_cup, y_cup, test_size=0.2, random_state=42
    )
    
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_test = scaler_x.transform(X_test_raw)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    # 2. CONFIG VELOCE (No Ensemble, meno epoche)
    fast_config = {
        'activation': 'ELU', 'lr': 0.005, 'weight_decay': 1e-4, 
        'dropout': 0.1, 'epochs': 400, 'batch_size': 128, # Batch più grande per velocità
        'optim': 'adam', 'use_scheduler': False, 'loss': 'Huber', 
        'patience': 30, 'es': True
    }
    
    # Architettura standard per il test (Deep Funnel medio)
    hidden_structure = [256, 128, 64] 

    # 3. DEFINIZIONE ORDINI DA TESTARE
    # Testiamo tutti gli ordini possibili? 4! = 24 combinazioni. Fattibile in ~10 minuti.
    targets = [0, 1, 2, 3]
    all_orders = list(itertools.permutations(targets))
    
    print(f"Testando {len(all_orders)} permutazioni diverse...")
    
    best_mee = float('inf')
    best_order = None
    
    results = []

    for idx, order in enumerate(all_orders):
        print(f"\n[{idx+1}/{len(all_orders)}] Testing Order: {order} ...")
        
        # Reset input per ogni ordine
        X_train_curr = X_train_t.clone()
        X_test_curr = X_test_t.clone()
        
        chain_preds = {}
        
        # Chain Loop veloce
        for target_idx in order:
            y_tr_single = y_train_t[:, target_idx:target_idx+1]
            # Valutazione interna rapida
            y_te_single_np = y_test_scaled[:, target_idx:target_idx+1]
            y_te_single = torch.FloatTensor(y_te_single_np).to(device)
            
            # Input dim cresce
            curr_input_dim = X_train_curr.shape[1]
            
            # Config specifica (se vuoi differenziare, ma qui teniamo uguale per velocità)
            cfg = fast_config.copy()
            cfg['hidden_layers'] = hidden_structure
            
            # Train singolo modello
            model, _, _ = train_model(cfg, curr_input_dim, X_train_curr, y_tr_single, X_test_curr, y_te_single, task_type='regression')
            
            # Predizioni (senza no_grad globale perché serve per il train loop, ma qui ok)
            model.eval()
            with torch.no_grad():
                p_train = model(X_train_curr)
                p_test = model(X_test_curr)
            
            # Salvataggio e Aggiornamento Feature
            chain_preds[target_idx] = p_test.cpu().numpy()
            
            # Aggiungi colonne
            X_train_curr = torch.cat([X_train_curr, p_train], dim=1)
            X_test_curr = torch.cat([X_test_curr, p_test], dim=1)
        
        # Calcolo MEE Globale per questo ordine
        final_preds = np.zeros((X_test.shape[0], 4))
        for i in range(4):
            final_preds[:, i:i+1] = chain_preds[i]
            
        diff = final_preds - y_test_scaled
        mee = np.mean(np.linalg.norm(diff, axis=1))
        
        print(f" -> MEE (Scaled): {mee:.5f}")
        results.append((order, mee))
        
        if mee < best_mee:
            best_mee = mee
            best_order = order
            print(f" ⭐ NUOVO BEST ORDER! {best_order}")

    print("\n" + "="*60)
    print(f"🏆 ORDINE VINCENTE: {best_order}")
    print(f"🏅 MEE RECORD (Fast Run): {best_mee:.5f}")
    print("="*60)
    
    print("\nTop 3 Ordini:")
    results.sort(key=lambda x: x[1])
    for i in range(3):
        print(f"{i+1}. {results[i][0]} - MEE: {results[i][1]:.5f}")

if __name__ == "__main__":
    main()
