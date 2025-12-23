import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', comment='#', header=None)

# 2. Selezione delle variabili
# Input: colonne da 1 a 12 (indici), escludendo x6 (indice 6)
# Target: y1 e y2 sono alle colonne 13 e 14 (indici)
input_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
X_raw = df.iloc[:, input_indices]
Y_raw = df.iloc[:, [13, 14, 15, 16]] 

# 3. Standardizzazione (Media 0, Varianza 1)
scaler_x = StandardScaler()
X_std = scaler_x.fit_transform(X_raw)

scaler_y = StandardScaler()
Y_std = scaler_y.fit_transform(Y_raw)

am_std_abs = np.mean(np.abs(X_std), axis=1).reshape(-1, 1)

norm_std = np.sqrt(np.sum(Y_std[:, 0:2]**2, axis=1))

model = LinearRegression()
model.fit(am_std_abs, norm_std)
norm_pred = model.predict(am_std_abs)

mee_val = mean_absolute_error(norm_std, norm_pred)

print(f"--- RISULTATI SU DATI STANDARDIZZATI ---")
print(f"MEE sulla norma R: {mee_val:.6f}")
print(f"Pendenza del modello (Slope): {model.coef_[0]:.6f}")
print(f"Intercetta: {model.intercept_:.6f}")

# Verifica della correlazione
correlation = np.corrcoef(am_std_abs.flatten(), norm_std)[0, 1]
print(f"Correlazione di Pearson: {correlation:.6f}")

target_diff = Y_std[:,2]-Y_std[:,3]
model = LinearRegression()
model.fit(am_std_abs, target_diff) 
target_pred = model.predict(am_std_abs)

# 7. Valutazione del Modello
mae = mean_absolute_error(target_diff, target_pred)
r2 = r2_score(target_diff, target_pred)

print("--- ANALISI REGRESSIONE PC1 vs (y4 - y3) ---")
print(f"Coefficiente di determinazione (R^2): {r2:.6f}")
print(f"Errore Medio Assoluto (MAE): {mae:.6f}")
print(f"Pendenza del modello (Slope): {model.coef_[0]:.6f}")
print(f"Intercetta: {model.intercept_:.6f}")
