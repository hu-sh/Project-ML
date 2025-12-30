import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', skiprows=7, header=None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler







X_raw = df.iloc[:, 1:13].values
y3 = df.iloc[:, 15].values
y4 = df.iloc[:, 16].values
y_diff = y3 - y4

# 2. Standardizzazione degli Input
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# 3. PCA per estrarre le prime 3 componenti
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)
pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]

# 4. Creazione del grafico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot: x=PC2, y=PC3, z=y_diff, colore=PC1
sc = ax.scatter(pc2, pc3, y_diff, c=pc1, cmap='magma', s=30, alpha=0.8)

ax.set_xlabel('PC2 (Variabile x6)')
ax.set_ylabel('PC3 (Variabile x5)')
ax.set_zlabel('Target 3 - Target 4')
ax.set_title('Analisi 3D: PC2, PC3 e Differenza Target\nColorato per Intensità di PC1')

# Aggiunta barra del colore
plt.colorbar(sc, label='Intensità PC1 (Segnale Latente)', shrink=0.5)

plt.show()
