from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gbrt import (
    evaluate_gbrt_regressor,
    grid_search_gbrt_regressor,
    train_gbrt_regressor,
)
from utils import load_cup_data


train_file = "data/CUP/ML-CUP25-TR.csv"

print(f"--- Training GBRT on {train_file} ---")

X_raw, y_raw = load_cup_data(train_file)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train_raw)
X_test = scaler_x.transform(X_test_raw)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

param_grid = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}

search_result = grid_search_gbrt_regressor(param_grid, X_train, y_train, k_folds=5)
best_model = search_result["best_model"]
best_config = search_result["best_config"]

print("Best GBRT config (train CV):", best_config)

metrics = evaluate_gbrt_regressor(best_model, X_train, y_train, X_test, y_test)

print(f"Training MSE: {metrics['train_mse']:.4f}")
print(f"Test MSE:     {metrics['test_mse']:.4f}")
print(f"Test RMSE:    {metrics['test_rmse']:.4f}")
print(f"Training MEE: {metrics['train_mee']:.4f}")
print(f"Test MEE:     {metrics['test_mee']:.4f}")
