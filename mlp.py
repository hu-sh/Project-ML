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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- 2. DEFINIZIONE MODELLO DINAMICO ---
class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_function):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_function())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


act_map = {"ReLU": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
def train_model(config, input_dim, X_train, y_train, X_val=None, y_val=None):
    h_layers = config['hidden_layers']
    act_fn = act_map[config['activation']]      
    lr = config['lr']
    mom = config['momentum']
    wd = config['weight_decay'] 
    epochs = config['epochs']
    batch_size = config.get('batch_size', 32) 
    
    model = DynamicMLP(input_dim, h_layers, act_fn) 
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss() 
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    
    model.train()
    train_loss_history = []
    val_loss_history = []



    patience = 20
    trigger_times = 0
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0) 
            
        # Loss media dell'epoca
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_loss_history.append(avg_loss)

        if X_val != None:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                v_loss = criterion(val_out, y_val)
                val_loss_history.append(v_loss.item())
            model.train()


            if v_loss < best_loss:
                best_loss = v_loss
                trigger_times = 0
                # Opzionale: salva qui i pesi del modello migliore
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    # print(f"Early stopping at epoch {epoch}")
                    break 
        
    return model, train_loss_history, val_loss_history


def grid_search_split(combinations, X, y):
    best_acc = 0
    best_config = {}
    best_model_state = None
    best_history = {}   
    best_model = None
    all_results = []     


    X_train, X_val, y_train, y_val = train_test_split(X_train_enc, y_train_, test_size=0.3, random_state=123)
    input_dim = X_train.shape[1]
    # Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)


    for i, config in enumerate(combinations):
 
        model, temp_train_loss, temp_val_loss = train_model(config, input_dim, X_train_t, y_train_t, X_val_t, y_val_t) 
            
        # Valutazione
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t)
            pred_lbl = (preds > 0.5).float()
            acc = (pred_lbl == y_val_t).sum() / len(y_val_t)
            acc_val = acc.item()

        
        config['acc'] = acc_val
        all_results.append(config)  
        
        # Salva il migliore
        if acc_val > best_acc or best_model == None:
            best_acc = acc_val
            best_config = config
            
            best_history = {
                'train': temp_train_loss[:], 
                'val': temp_val_loss[:]
            }

            best_model = model
            # Salva i pesi (state_dict) se vuoi riusarlo
            # torch.save(model.state_dict(), 'best_model_temp.pth')


    return best_acc, best_config, best_history, all_results


def grid_search_kfold_cv(combinations, X, y):
    best_acc = 0
    best_std = -1
    best_config = {}
    best_model_state = None
    best_history = {}   
    best_model = None
    all_results = []   

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)


    for i, config in enumerate(combinations):
     
        val_accuracies = []
        
        # --- INIZIO CROSS VALIDATION PER QUESTA CONFIG ---
        # skf.split vuole X e y per bilanciare le classi, anche se usiamo gli indici
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # A. Split dei dati raw
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            
            # C. Conversione in Tensori
            X_tr_t = torch.FloatTensor(X_train_fold)
            y_tr_t = torch.FloatTensor(y_train_fold).view(-1, 1)
            X_val_t = torch.FloatTensor(X_val_fold)
            y_val_t = torch.FloatTensor(y_val_fold).view(-1, 1)
            
            input_dim = X_tr_t.shape[1]
            
            model, tmp_train_loss, tmp_val_loss = train_model(config, input_dim, X_tr_t, y_tr_t, X_val_t, y_val_t)
                
            # F. Valutazione sul Fold corrente
            model.eval()
            with torch.no_grad():
                preds = model(X_val_t)
                preds_lbl = (preds > 0.5).float()
                acc = (preds_lbl == y_val_t).float().mean().item()
                val_accuracies.append(acc)
                
        # --- FINE CROSS VALIDATION ---
        
        # Calcola media accuracy per questa config
        avg_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        print(f"Config {i}: Avg Acc: {avg_acc:.4f} (+/- {std_acc:.4f}) | {config}")
        
        # Salva risultato
        res_entry = config.copy()
        res_entry['mean_val_acc'] = avg_acc
        res_entry['std_val_acc'] = std_acc
        all_results.append(res_entry)
        
        if avg_acc > best_acc or best_model == None:
            best_acc = avg_acc 
            best_std = std_acc
            best_config = config
            
            best_history = {
                'train': tmp_train_loss[:], 
                'val': tmp_val_loss[:]
            }

            # Salva i pesi (state_dict) se vuoi riusarlo
            # torch.save(model.state_dict(), 'best_model_temp.pth')


    return best_acc, best_std, best_config, best_history, all_results



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

X_test_t = torch.FloatTensor(X_test_enc)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

   

# --- 3. GRIGLIA IPERPARAMETRI ---
param_grid = {
    'hidden_layers': [[5], [10], [20], [10, 10]], 
    'activation': ["ReLU", "tanh"], 
    'lr': [0.1, 0.01, 0.001], 
    'momentum': [0.7, 0.9], 
    'weight_decay': [0.0001, 0.001], 
    'epochs': [500] 
}

# Genera tutte le combinazioni
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Combinations to test: {len(combinations)}")

best_acc, best_std, best_config, best_history, all_results = grid_search_kfold_cv(combinations, X_train_enc, y_train)
df = pd.DataFrame(all_results)




print("\n-------------------------------------------")
print(f"🏆 Best Configuration Found (Acc: {best_acc:.4f}):")
print(f"Hidden layers: {best_config['hidden_layers']}")
print(f"Activation function: {best_config['activation']}")
print(f"LR: {best_config['lr']} | Momentum: {best_config['momentum']}")
print(f"Weight decay: {best_config['weight_decay']}")
print(f"Epochs: {best_config['epochs']}")
print("-------------------------------------------")



############ BEST CONFIG TEST ON MONKS.TEST
X = torch.FloatTensor(X_train_enc)
y = torch.FloatTensor(y_train).view(-1, 1)
model, train_loss, val_loss = train_model(best_config, input_dim, X, y)
model.eval()
with torch.no_grad():
    preds = model(X_test_t)
    pred_lbl = (preds > 0.5).float()
    acc = (pred_lbl == y_test_t).sum() / len(y_test_t)
    acc_test = acc.item()
    print("Accuracy on test set: ", acc_test)



## cambiare il random-state dello splitting porta a risultati diversi. fare k-fold cv dovrebbe risolvere.



