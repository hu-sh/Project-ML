import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from poly import evaluate_poly_regressor, grid_search_poly_regressor, train_poly_regressor
from utils import load_cup_data


def main() -> None:
    X, y = load_cup_data("data/CUP/ML-CUP25-TR.csv")

    # Scale targets so metrics are reported on standardized variables
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    param_grid = {
        "degree": [1, 2, 3, 4, 5, 6, 7, 8],
        "alpha": [5.0, 50.0, 500.0, 5000.0, 50000.0],
        "include_bias": [False],
    }

    print("Starting polynomial Ridge grid search on CUP...")
    search_results = grid_search_poly_regressor(
        param_grid=param_grid,
        X=X,
        y=y_scaled,
        k_folds=5,
        random_state=0,
    )

    best_config = search_results["best_config"]
    print("\nBest configuration (CV):", best_config)

    # Hold-out evaluation on the best configuration for a quick sanity check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=0
    )
    best_model = train_poly_regressor(X_train, y_train, **best_config)
    metrics = evaluate_poly_regressor(best_model, X_train, y_train, X_test, y_test)

    print("\nHold-out evaluation with best config (on scaled targets):")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Train MEE : {metrics['train_mee']:.4f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test  MEE : {metrics['test_mee']:.4f}")
    print(f"Test  RMSE per target: {np.array(metrics['test_rmse_per_target'])}")


if __name__ == "__main__":
    main()
