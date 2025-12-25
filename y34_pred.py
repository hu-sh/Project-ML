import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE
# ==========================================
CSV_PATH = 'data/CUP/ML-CUP25-TR.csv'

try:
    df = pd.read_csv(CSV_PATH,comment='#', header=None)
    # Input: Col 1-12
    X = df.iloc[:, [1,2,3,4,5,7,8,9,10,11,12]].values
    # Target: y3 - y4
    y = (df.iloc[:, -2] - df.iloc[:, -1]).values
    
    # AGGIUNTA MANUALE DEL QUADRATO (come avevi chiesto)
    # Supponiamo che x6 sia all'indice 5. Aggiungiamo (y3-y4)^2 è impossibile 
    # perché è il target, ma possiamo aggiungere i quadrati degli input.
    # Se vuoi il quadrato della differenza degli input, lo calcoliamo qui:
    # Esempio: aggiungiamo una feature che è il quadrato della colonna 5
    x6_squared = X[:, 5:6]**2
    X = np.hstack((X, x6_squared))
    
    print(f"Dataset caricato. Feature totali (incluse derivate): {X.shape[1]}")
except Exception as e:
    print(f"Errore: {e}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione (fondamentale per MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. ENSEMBLE: BOOSTING DI MLP LINEARI
# ==========================================

# Definiamo il modello base: MLP SENZA HIDDEN LAYER
# hidden_layer_sizes=() significa collegamento diretto Input -> Output
base_mlp = MLPRegressor(
    hidden_layer_sizes=(), 
    activation='identity', # Forza la linearità
    solver='lbfgs', 
    max_iter=500,
    random_state=42
)

# Applichiamo AdaBoost
# n_estimators: quanti MLP lineari addestriamo in sequenza
# learning_rate: quanto "pesa" la correzione di ogni modello successivo
boosted_mlp = AdaBoostRegressor(
    estimator=base_mlp,
    n_estimators=1000,
    learning_rate=0.01,
    loss='square',
    random_state=42
)

print("\nAddestramento Boosting di MLP Lineari...")
boosted_mlp.fit(X_train_scaled, y_train)

# ==========================================
# 3. VALUTAZIONE
# ==========================================
y_pred = boosted_mlp.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"RISULTATI BOOSTED MLP (No Hidden Layers):")
print(f"RMSE: {rmse:.5f}")
print(f"R2 Score: {r2:.5f}")
print("-" * 30)

# Visualizzazione residui
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, y_test - y_pred, alpha=0.5, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predizioni')
plt.ylabel('Residui (Errore)')
plt.title('Analisi dei Residui - Boosted MLP')
plt.show()
