import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

# Carica
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', skiprows=7, header=None)
X = df.iloc[:, 1:13].values
Y = df.iloc[:, 13:17].values

# Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scala
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

# Fit NN Max
nbrs = NearestNeighbors(n_neighbors=4).fit(X_train_sc)
dist, idx = nbrs.kneighbors(X_val_sc)

# Predict Loop
p2, p3, p4 = [], [], []
for i in range(len(X_val)):
    # K=2
    m2 = Ridge(alpha=1e-5).fit(X_train[idx[i, :2]], Y_train[idx[i, :2]])
    p2.append(m2.predict([X_val[i]])[0]) # Nota: predict su raw features se fit su raw
    
    # K=3
    m3 = Ridge(alpha=1e-5).fit(X_train[idx[i, :3]], Y_train[idx[i, :3]])
    p3.append(m3.predict([X_val[i]])[0])
    
    # K=4
    m4 = Ridge(alpha=1e-5).fit(X_train[idx[i, :4]], Y_train[idx[i, :4]])
    p4.append(m4.predict([X_val[i]])[0])

p2, p3, p4 = np.array(p2), np.array(p3), np.array(p4)
ens = (p2 + p3 + p4) / 3.0

mee = np.mean(np.linalg.norm(ens - Y_val, axis=1))
print(f"MEE Validation: {mee:.5f}")
