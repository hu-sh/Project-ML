import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from utils import load_monk_data, get_encoder
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_function, output_size=1, task_type='regression'):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layer = nn.Linear(in_dim, h_dim)
            
            if activation_function == nn.Tanh:
                nn.init.xavier_uniform_(layer.weight)
            elif activation_function == nn.ReLU or activation_function == nn.LeakyReLU: 
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')            

            layers.append(layer)
            layers.append(activation_function())
            
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_size))
        
        if task_type == 'classification':
            layers.append(nn.Sigmoid())
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def average_histories(histories):
    averaged = {}
    metrics = histories[0].keys() # es. ['train_loss', 'val_loss', ...]

    max_epochs = max([len(h['train_loss']) for h in histories])

    for metric in metrics:
        all_fold_values = []

        for h in histories:
            values = h[metric] 
            current_len = len(values)

            if current_len < max_epochs:
                diff = max_epochs - current_len
                
                last_val = values[-1]
                
                padding = np.full(diff, last_val)
                
                values = np.concatenate((values, padding))
            
            all_fold_values.append(values)

        all_fold_stack = np.vstack(all_fold_values)

        averaged[metric] = np.mean(all_fold_stack, axis=0) # es. 'val_loss' diventa la media
        averaged[f'std_{metric}'] = np.std(all_fold_stack, axis=0)

    return averaged


act_map = {"ReLU": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "LeakyReLU": nn.LeakyReLU}
def train_model(config, input_dim, X_train, y_train, X_val=None, y_val=None, task_type='regression'):
    # X_train, y_train, X_val, y_val must be tensorized
    h_layers = config['hidden_layers']
    act_fn = act_map[config['activation']]      
    lr = config['lr']
    mom = config['momentum']
    wd = config['weight_decay'] 
    epochs = config['epochs']
    batch_size = config.get('batch_size', 32) 
    es = config.get('es', False)
    
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = DynamicMLP(input_dim, h_layers, act_fn, output_size=output_dim, task_type=task_type)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    model.train()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_score': [], 'val_score': []
    }


    patience = 20
    trigger_times = 0
    best_loss = float('inf')
    stop_epoch = epochs
    
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
        history['train_loss'].append(avg_loss)

        model.eval()
        with torch.no_grad():
            train_out = model(X_train)

            if task_type == 'regression':
                tr_errors = train_out - y_train
                train_score  = torch.norm(tr_errors, p=2, dim=1).mean().item()
            else:
                train_pred = (train_out > 0.5).float()
                train_score = (train_pred == y_train).float().mean().item()
            

            history['train_score'].append(train_score)

            # Validation
            if X_val is not None:
                val_out = model(X_val)
                v_loss = criterion(val_out, y_val)
                
                if task_type == 'regression':
                    # MEE per Regressione
                    val_err = val_out - y_val
                    val_score = torch.norm(val_err, p=2, dim=1).mean().item()
                else:
                    # Accuracy per Classificazione
                    val_pred = (val_out > 0.5).float()
                    val_score = (val_pred == y_val).float().mean().item()               
                
                history['val_loss'].append(v_loss.item())
                history['val_score'].append(val_score)

                # Early Stopping Check
                if v_loss < best_loss:
                    best_loss = v_loss
                    trigger_times = 0
                else:
                    trigger_times += 1
                    
                    if es and trigger_times >= patience:
                        stop_epoch = epoch
                        break 
    for key in history:
        history[key] = np.array(history[key])
    
    return model, history, stop_epoch




def plot_training_history(history, title=""):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(16, 6))
    
    # --- GRAFICO LOSS ---
    plt.subplot(1, 2, 1)
    
    # Training Loss
    tr_loss = history['train_loss']
    std_tr_loss = history.get('std_train_loss', np.zeros(len(tr_loss)))
    plt.plot(epochs, tr_loss, label='Training Loss', color='blue')
    plt.fill_between(epochs, 
                     tr_loss - std_tr_loss, 
                     tr_loss + std_tr_loss, 
                     color='blue', alpha=0.15, label='Std Dev (Train)')
    
    # Validation Loss
    if 'val_loss' in history:
        val_loss = history['val_loss']
        std_val_loss = history.get('std_val_loss', np.zeros(len(val_loss)))       
        plt.plot(epochs, val_loss, label='Avg Validation Loss', color='orange', linestyle='--')
        plt.fill_between(epochs, 
                         val_loss - std_val_loss, 
                         val_loss + std_val_loss, 
                         color='orange', alpha=0.15, label='Std Dev (Val)')
    
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # --- GRAFICO ACCURACY ---
    plt.subplot(1, 2, 2)
    
    # Training Accuracy
    if 'train_score' in history:
        tr_score = history['train_score']
        std_tr_score = history.get('std_train_score', np.zeros(len(tr_score)))
        plt.plot(epochs, tr_score, label='Avg Training Acc', color='green')
        plt.fill_between(epochs, 
                         tr_score - std_tr_score, 
                         tr_score + std_tr_score, 
                         color='green', alpha=0.15, label='Std Dev (Train)')

    # Validation Accuracy
    if 'val_score' in history:
        val_score = history['val_score']
        std_val_score = history.get('std_val_score', np.zeros(len(val_score)))
        plt.plot(epochs, val_score, label='Avg Validation Acc', color='red', linestyle='--')
        plt.fill_between(epochs, 
                         val_score - std_val_score, 
                         val_score + std_val_score, 
                         color='red', alpha=0.15, label='Std Dev (Val)')
    
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def grid_search_split(combinations, X, y, task_type='regression'):
    best_score = float('inf') if task_type=='regression' else -1
    best_config = {}
    best_history = {}   

    X_train, X_val, y_train, y_val = train_test_split(X_train_enc, y_train_, test_size=0.3, random_state=123)
    input_dim = X_train.shape[1]
    # Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    if task_type == 'classification':
        y_train_t = y_train_t.view(-1, 1)
        y_val_t = y_val_t.view(-1,1)

    for i, config in enumerate(combinations):
 
        model, history, _  = train_model(config, input_dim, X_train_t, y_train_t, X_val_t, y_val_t, task_type=task_type) 
            
        all_results.append(config)  
        
        # Salva il migliore
        if (task_type=='regression') == (history['val_score'][-1] < best_score):
            best_score = score_val
            best_config = config
            
            best_history = history

            # Salva i pesi (state_dict) se vuoi riusarlo
            # torch.save(model.state_dict(), 'best_model_temp.pth')


    return best_config, best_history, all_results


def grid_search_kfold_cv(combinations, k_folds, X, y, task_type='regression'):
    best_score = float('inf') if task_type=='regression' else -1
    best_std = -1
    best_config = {}
    best_histories = {}  
    best_avg_stop = -1
    all_results = []   

    if task_type == 'regression':
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for i, config in enumerate(combinations):
     
        val_scores = []
        histories = []
        stops = 0

        # --- INIZIO CROSS VALIDATION PER QUESTA CONFIG ---
        # skf.split vuole X e y per bilanciare le classi, anche se usiamo gli indici
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # A. Split dei dati raw
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            
            # C. Conversione in Tensori
            X_tr_t = torch.FloatTensor(X_train_fold)
            y_tr_t = torch.FloatTensor(y_train_fold)
            X_val_t = torch.FloatTensor(X_val_fold)
            y_val_t = torch.FloatTensor(y_val_fold)

            if task_type == 'classification':
                y_tr_t = y_tr_t.view(-1, 1)
                y_val_t = y_val_t.view(-1,1)

            input_dim = X_tr_t.shape[1]
            
            model, history, stop = train_model(config, input_dim, X_tr_t, y_tr_t, X_val_t, y_val_t, task_type=task_type)
            stops += stop
            histories.append(history)

            val_scores.append(history['val_score'][-1])
                
        # --- FINE CROSS VALIDATION ---
        
        # Calcola media scoreuracy per questa config
        avg_score = np.mean(val_scores)
        std_score = np.std(val_scores)
        
        score_string = 'scoreuracy' if task_type == 'classification' else 'MEE'
        print(f"Config {i}: Avg {score_string}: {avg_score:.4f} (+/- {std_score:.4f}) | {config}")
        
        # Salva risultato
        res_entry = config.copy()
        res_entry['mean_val_score'] = avg_score
        res_entry['std_val_score'] = std_score
        all_results.append(res_entry)
        
        if (task_type=='regression') == (avg_score < best_score):
            best_score = avg_score 
            best_config = config
            
            best_histories = histories

            best_avg_stop = stops/k_folds

            # Salva i pesi (state_dict) se vuoi riusarlo
            # torch.save(model.state_dict(), 'best_model_temp.pth')
            # torch.save(model.state_dict(), 'best_model_temp.pth')


    return best_config, best_histories, best_avg_stop, all_results





