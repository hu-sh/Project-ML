from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from gbrt import (
    evaluate_gbrt_regressor,
    grid_search_gbrt_regressor,
    train_gbrt_regressor,
)
from utils import load_cup_data


def build_extra_features(X: np.ndarray) -> np.ndarray:
    """Construct selected engineered features to augment the raw inputs."""
    eps = 1e-12
    sign_input_10 = np.sign(X[:, 9]).reshape(-1, 1)
    meanabs_all_input = np.mean(np.abs(X), axis=1, keepdims=True)
    radius_input_3_7 = np.hypot(X[:, 2], X[:, 6]).reshape(-1, 1)
    radius_input_7_10 = np.hypot(X[:, 6], X[:, 9]).reshape(-1, 1)
    radius_input_3_10 = np.hypot(X[:, 2], X[:, 9]).reshape(-1, 1)
    radius_input_3_9 = np.hypot(X[:, 2], X[:, 8]).reshape(-1, 1)
    diff_input_7_10 = (X[:, 6] - X[:, 9]).reshape(-1, 1)
    diff_input_3_10 = (X[:, 2] - X[:, 9]).reshape(-1, 1)
    diff_input_1_10 = (X[:, 0] - X[:, 9]).reshape(-1, 1)
    diff_input_9_10 = (X[:, 8] - X[:, 9]).reshape(-1, 1)

    return np.hstack(
        [
            sign_input_10,
            meanabs_all_input,
            radius_input_3_7,
            radius_input_7_10,
            radius_input_3_10,
            radius_input_3_9,
            diff_input_7_10,
            diff_input_3_10,
            diff_input_1_10,
            diff_input_9_10,
        ]
    )


def main():
    train_file = "data/CUP/ML-CUP25-TR.csv"
    print(f"--- Training GBRT with engineered features on {train_file} ---")

    X_raw, y_raw = load_cup_data(train_file)
    y_main = y_raw[:, :4]  # predict y1..y4 only

    X_extra = build_extra_features(X_raw)
    X_feat = np.hstack([X_raw, X_extra])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_feat, y_main, test_size=0.2, random_state=42
    )

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_test = scaler_x.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    search_result = grid_search_gbrt_regressor(param_grid, X_train, y_train_scaled, k_folds=3)
    best_model = search_result["best_model"]
    best_config = search_result["best_config"]

    print("Best GBRT config (train CV):", best_config)

    metrics = evaluate_gbrt_regressor(best_model, X_train, y_train_scaled, X_test, y_test_scaled)

    print(f"Training MSE: {metrics['train_mse']:.4f}")
    print(f"Test MSE:     {metrics['test_mse']:.4f}")
    print(f"Test RMSE:    {metrics['test_rmse']:.4f}")
    print(f"Training MEE: {metrics['train_mee']:.4f}")
    print(f"Test MEE:     {metrics['test_mee']:.4f}")

    # Report MAE per output (y1..y4) in original scale
    y_test_inv = scaler_y.inverse_transform(y_test_scaled)
    y_pred_inv = scaler_y.inverse_transform(best_model.predict(X_test))
    mae_main = np.mean(np.abs(y_pred_inv - y_test_inv), axis=0)
    rmse_main = np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2))

    print("MAE per output (y1..y4):", np.round(mae_main, 4))
    print(f"RMSE complessivo (y1..y4): {rmse_main:.4f}")


if __name__ == "__main__":
    main()
