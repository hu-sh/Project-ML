import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from sklearn.model_selection import train_test_split


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


X_train, X_val, y_train, y_val = train_test_split(
    X_std, Y_std, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train[:, 2]-y_train[:, 3])
y_pred = model.predict(X_val)

mee_val = mean_absolute_error(y_val[:,2]-y_val[:,3], y_pred)

print(f"--- RISULTATI SU DATI STANDARDIZZATI ---")
print(f"MEE sulla norma R: {mee_val:.6f}")
print(f"Pendenza del modello (Slope): {model.coef_[0]:.6f}")
print(f"Intercetta: {model.intercept_:.6f}")


