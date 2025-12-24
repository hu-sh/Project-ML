import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import grid_search_kfold_cv, train_model, device
from pca import grid_search_pca_poly_regressor
from utils import load_cup_data


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mee(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_true - y_pred).reshape(len(y_true), -1)
    return float(np.linalg.norm(diff, axis=1).mean())


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mee": _mee(y_true, y_pred),
        "rmse_per_target": np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)),
    }


MLP_CONFIGS = {
    0: {
        "hidden_layers": [512, 256, 128],
        "activation": "LeakyReLU",
        "lr": 0.005,
        "dropout": 0.1,
        "loss": "MSE",
        "weight_decay": 0.0001,
        "epochs": 600,
        "batch_size": 64,
        "es": True,
        "patience": 40,
        "use_scheduler": True,
    },
    1: {
        "hidden_layers": [512, 256, 128],
        "activation": "LeakyReLU",
        "lr": 0.005,
        "dropout": 0.1,
        "loss": "MSE",
        "weight_decay": 0.0001,
        "epochs": 600,
        "batch_size": 64,
        "es": True,
        "patience": 40,
        "use_scheduler": True,
    },
    2: {
        "hidden_layers": [512, 256, 128],
        "activation": "LeakyReLU",
        "lr": 0.001,
        "dropout": 0.1,
        "loss": "MSE",
        "weight_decay": 0.0001,
        "epochs": 600,
        "batch_size": 64,
        "es": True,
        "patience": 40,
        "use_scheduler": True,
    },
    3: {
        "hidden_layers": [512, 256, 128],
        "activation": "LeakyReLU",
        "lr": 0.005,
        "dropout": 0.2,
        "loss": "MSE",
        "weight_decay": 0.0001,
        "epochs": 600,
        "batch_size": 64,
        "es": True,
        "patience": 40,
        "use_scheduler": True,
    },
}


def train_mlp_models(X_train: np.ndarray, y_train: np.ndarray) -> list:
    """Run a tiny grid (provided configs) per target and fit one MLP per output."""
    combos = list(MLP_CONFIGS.values())
    mlp_models = []

    # small validation split shared across targets for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=0
    )

    for target_idx in range(y_train.shape[1]):
        print(f"\n=== MLP target {target_idx} ===")
        y_single_tr = y_tr[:, target_idx : target_idx + 1]
        y_single_val = y_val[:, target_idx : target_idx + 1]

        best_config, _, _, _ = grid_search_kfold_cv(
            combos, k_folds=5, X=X_tr, y=y_single_tr, task_type="regression"
        )
        print(f"Best MLP config for target {target_idx}: {best_config}")

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_single_tr)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_single_val)

        model, _, stop_epoch = train_model(
            best_config,
            input_dim=X_train.shape[1],
            X_train=X_tr_t,
            y_train=y_tr_t,
            X_val=X_val_t,
            y_val=y_val_t,
            task_type="regression",
        )
        print(f"Finished training target {target_idx} at epoch {stop_epoch}")
        mlp_models.append(model)

    return mlp_models


def predict_mlp(models: list, X: np.ndarray) -> np.ndarray:
    preds = []
    X_t = torch.FloatTensor(X).to(device)
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        preds.append(pred)
    return np.hstack(preds)


def main() -> None:
    X_raw, y_raw = load_cup_data("data/CUP/ML-CUP25-TR.csv")

    # Train/test split for hold-out evaluation
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=0
    )

    # Scaling for MLP branch
    x_scaler = StandardScaler().fit(X_train_raw)
    X_train_mlp = x_scaler.transform(X_train_raw)
    X_test_mlp = x_scaler.transform(X_test_raw)

    y_scaler = StandardScaler().fit(y_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)

    # ---- PCA + Poly Ridge branch (uses raw X, scaled y) ----
    pca_param_grid = {
        "n_components": [2, 3, 4, 5],
        "degree": [1, 2, 3, 4, 5, 6, 7, 8],
        "alpha": [5.0, 50.0, 500.0, 5000.0, 50000.0],
        "include_bias": [False],
        "pca_whiten": [False],
    }

    print("\n=== Grid search: PCA + Poly Ridge ===")
    pca_results = grid_search_pca_poly_regressor(
        param_grid=pca_param_grid,
        X=X_train_raw,
        y=y_train_scaled,
        k_folds=5,
        random_state=0,
    )
    pca_best_model = pca_results["best_model"]
    print("Best PCA+Poly config:", pca_results["best_config"])

    # ---- MLP branch ----
    print("\n=== Grid search + train: MLP (fixed configs) ===")
    mlp_models = train_mlp_models(X_train_mlp, y_train_scaled)

    # ---- Predictions ----
    pca_preds = pca_best_model.predict(X_test_raw)
    mlp_preds = predict_mlp(mlp_models, X_test_mlp)
    hybrid_preds = 0.5 * (pca_preds + mlp_preds)

    # ---- Metrics (on scaled targets) ----
    pca_metrics = evaluate_predictions(y_test_scaled, pca_preds)
    mlp_metrics = evaluate_predictions(y_test_scaled, mlp_preds)
    hybrid_metrics = evaluate_predictions(y_test_scaled, hybrid_preds)

    def _fmt(metrics: dict) -> str:
        rmse_pt = metrics["rmse_per_target"]
        return (
            f"RMSE: {metrics['rmse']:.4f} | MEE: {metrics['mee']:.4f} | "
            f"RMSE per target: {rmse_pt}"
        )

    print("\n=== Hold-out metrics (scaled targets) ===")
    print("PCA+Poly Ridge :", _fmt(pca_metrics))
    print("MLP            :", _fmt(mlp_metrics))
    print("Hybrid (avg)   :", _fmt(hybrid_metrics))


if __name__ == "__main__":
    main()
