import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from sklearn.model_selection import train_test_split
from utils import load_monk_data, get_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SETUP DATI ---
filename = 'data/MONK/monks-2.train' # Prova difficile
X_raw, y = load_monk_data(filename)
encoder = get_encoder(X_raw)
X_encoded = encoder.transform(X_raw)

# TODO 
# a better split for training/validation
# Split 70/30 per la selezione
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
INPUT_DIM = X_train.shape[1]

# Tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)

# --- 2. DEFINIZIONE MODELLO DINAMICO ---
class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_class):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_class())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def grid_search(combinations):
    best_acc = 0
    best_config = {}
    best_model_state = None
    best_history = {}   
    all_results = []      

    for i, config in enumerate(combinations):
        h_layers = config['hidden_layers']
        act_cls = nn.ReLu if config['activation'] == "ReLu" else nn.Tanh
        lr = config['lr']
        mom = config['momentum']
        wd = config['weight_decay'] 
        epochs = config['epochs']

        # Crea modello nuovo (TABULA RASA)
        model = DynamicMLP(input_size=INPUT_DIM, hidden_layers=h_layers, activation_class=act_cls)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
        criterion = nn.MSELoss()
        
        temp_train_loss = []
        temp_val_loss = []    
       
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            temp_train_loss.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                v_loss = criterion(val_out, y_val_t)
                temp_val_loss.append(v_loss.item())
            model.train() 
            
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
        if acc_val > best_acc:
            best_acc = acc_val
            best_config = config
            
            best_history = {
                'train': temp_train_loss[:], 
                'val': temp_val_loss[:]
            }
            # Salva i pesi (state_dict) se vuoi riusarlo
            # torch.save(model.state_dict(), 'best_model_temp.pth')


        return best_acc, best_config, best_history, all_results



# --- 3. GRIGLIA IPERPARAMETRI ---
param_grid = {
    'hidden_layers': [[10], [20], [10, 10]], # [10]=1 layer, [10,10]=2 layer
    'activation': ["ReLu", "tanh" ],        
    'lr': [0.1, 0.05, 0.01, 0.001],
    'momentum': [0.9, 0.8, 0.5, 0],
    'weight_decay': [0.01, 0.0001, 0], 
    'epochs': [100, 300]
}

# Genera tutte le combinazioni
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Combinations to test: {len(combinations)}")

best_acc, best_config, best_history, all_results = grid_search(combinations)
df = pd.DataFrame(all_results)


print("\n-------------------------------------------")
print(f"🏆 Best Configuration Found (Acc: {best_acc:.4f}):")
print(f"Hideden layers: {best_config['hidden_layers']}")
print(f"Activation function: {best_config['activation_name']}")
print(f"LR: {best_config['lr']} | Momentum: {best_config['momentum']}")
print("-------------------------------------------")
