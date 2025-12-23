from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from gbrt import (
    evaluate_gbrt_regressor,
    grid_search_gbrt_regressor,
    train_gbrt_regressor,
)
from utils import load_cup_data


train_file = "data/CUP/ML-CUP25-TR.csv"

print(f"--- Training GBRT (stacking aux -> main) on {train_file} ---")

X_raw, y_raw_orig = load_cup_data(train_file)

# Targets
y_main = y_raw_orig[:, :4]
y_r12 = np.hypot(y_raw_orig[:, 0], y_raw_orig[:, 1]).reshape(-1, 1)
y_diff34 = (y_raw_orig[:, 2] - y_raw_orig[:, 3]).reshape(-1, 1)
y_meanabs = np.mean(np.abs(y_main), axis=1, keepdims=True)
y_aux = np.hstack([y_r12, y_diff34, y_meanabs])  # shape (n,3)

# Train/test split
X_train_raw, X_test_raw, y_train_main, y_test_main, y_train_aux, y_test_aux = train_test_split(
    X_raw, y_main, y_aux, test_size=0.2, random_state=42
)

# Scale inputs for aux model
scaler_x_aux = StandardScaler()
X_train_aux = scaler_x_aux.fit_transform(X_train_raw)
X_test_aux = scaler_x_aux.transform(X_test_raw)

# Aux model: predict r12, diff34, meanabs
param_grid_aux = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}
search_aux = grid_search_gbrt_regressor(param_grid_aux, X_train_aux, y_train_aux, k_folds=3)
aux_model = search_aux["best_model"]
aux_config = search_aux["best_config"]
print("Best aux GBRT config:", aux_config)

aux_pred_train = aux_model.predict(X_train_aux)
aux_pred_test = aux_model.predict(X_test_aux)

# Build features for main model: X plus aux predictions
X_train_aug_raw = np.hstack([X_train_raw, aux_pred_train])
X_test_aug_raw = np.hstack([X_test_raw, aux_pred_test])

scaler_x_main = StandardScaler()
X_train_main = scaler_x_main.fit_transform(X_train_aug_raw)
X_test_main = scaler_x_main.transform(X_test_aug_raw)

scaler_y_main = StandardScaler()
y_train_main_scaled = scaler_y_main.fit_transform(y_train_main)
y_test_main_scaled = scaler_y_main.transform(y_test_main)

param_grid_main = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}

search_result = grid_search_gbrt_regressor(param_grid_main, X_train_main, y_train_main_scaled, k_folds=3)
best_model = search_result["best_model"]
best_config = search_result["best_config"]

print("Best main GBRT config (train CV):", best_config)

metrics = evaluate_gbrt_regressor(best_model, X_train_main, y_train_main_scaled, X_test_main, y_test_main_scaled)

print(f"Training MSE: {metrics['train_mse']:.4f}")
print(f"Test MSE:     {metrics['test_mse']:.4f}")
print(f"Test RMSE:    {metrics['test_rmse']:.4f}")
print(f"Training MEE: {metrics['train_mee']:.4f}")
print(f"Test MEE:     {metrics['test_mee']:.4f}")

# Report MAE per output (y1..y4) in original scale
y_test_inv = scaler_y_main.inverse_transform(y_test_main_scaled)
y_pred_inv = scaler_y_main.inverse_transform(best_model.predict(X_test_main))
mae_main = np.mean(np.abs(y_pred_inv - y_test_inv), axis=0)
rmse_main = np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2))

print("MAE per output (y1..y4):", np.round(mae_main, 4))
print(f"RMSE complessivo (y1..y4): {rmse_main:.4f}")
