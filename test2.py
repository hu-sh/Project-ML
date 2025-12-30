import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

# 1. Caricamento Dati
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', skiprows=7, header=None)
X = df.iloc[:, 1:13].values
Y = df.iloc[:, 13:17].values

# 2. Split Rigoroso (Training / Validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Preprocessing
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

# 4. Manifold Learning (PCA)
# Usiamo solo le prime 4 componenti per trovare i vicini "veri" sul manifold
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca = pca.transform(X_val_sc)

# 5. Local Learning (Hybrid)
# Distanza calcolata su PCA (4 dim), Regressione su Originali (12 dim)
k_neighbors = 2
nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(X_train_pca)
distances, indices = nbrs.kneighbors(X_val_pca)

preds = []
for i in range(len(X_val)):
    idx = indices[i]
    
    # Prendi input e target dei vicini
    X_loc = X_train_sc[idx]
    Y_loc = Y_train[idx]
    
    # Regressione Ridge Locale
    model = Ridge(alpha=1e-5)
    model.fit(X_loc, Y_loc)
    
    # Predici usando l'input originale standardizzato
    preds.append(model.predict([X_val_sc[i]])[0])

Y_pred = np.array(preds)

# 6. Valutazione
mee = np.mean(np.linalg.norm(Y_pred - Y_val, axis=1))
print(f"--------------------------------------------------")
print(f"Validazione PCA-Manifold (PCA=4, K=2)")
print(f"MEE Validation Set: {mee:.5f}")
print(f"--------------------------------------------------")
