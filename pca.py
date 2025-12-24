from itertools import product
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_pca_poly_regressor(
    n_components: int = 3,
    degree: int = 2,
    alpha: float = 1.0,
    include_bias: bool = False,
    pca_whiten: bool = False,
    **ridge_kwargs: Any,
) -> Pipeline:
    """Create a PCA -> PolynomialFeatures -> Ridge pipeline."""
    return Pipeline(
        [
            ("xscale", StandardScaler()),
            ("pca", PCA(n_components=n_components, whiten=pca_whiten, random_state=0)),
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("ridge", Ridge(alpha=alpha, **ridge_kwargs)),
        ]
    )


def train_pca_poly_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **config: Any,
) -> Pipeline:
    """Fit the PCA+poly Ridge regressor with the provided configuration."""
    model = build_pca_poly_regressor(**config)
    model.fit(X_train, y_train)
    return model


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mee(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_true - y_pred).reshape(len(y_true), -1)
    return float(np.linalg.norm(diff, axis=1).mean())


def _per_target_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def evaluate_pca_poly_regressor(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute train/test RMSE and MEE for a fitted model."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_rmse": _rmse(y_train, train_pred),
        "test_rmse": _rmse(y_test, test_pred),
        "train_mee": _mee(y_train, train_pred),
        "test_mee": _mee(y_test, test_pred),
        "test_rmse_per_target": _per_target_rmse(y_test, test_pred).tolist(),
    }


def _expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def grid_search_pca_poly_regressor(
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    random_state: int = 0,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """K-Fold grid search minimizing RMSE for PCA+poly Ridge regressors."""
    combos = _expand_param_grid(param_grid)
    cv = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state)

    best_config: Dict[str, Any] = {}
    best_score = np.inf
    cv_results: List[Dict[str, Any]] = []

    for idx, config in enumerate(combos, start=1):
        rmses: List[float] = []
        for train_idx, val_idx in cv.split(X):
            model = train_pca_poly_regressor(X[train_idx], y[train_idx], **config)
            val_pred = model.predict(X[val_idx])
            rmses.append(_rmse(y[val_idx], val_pred))

        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))

        result_entry = config.copy()
        result_entry.update({"mean_val_rmse": mean_rmse, "std_val_rmse": std_rmse})
        cv_results.append(result_entry)

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_config = config

        print(
            f"[{idx}/{len(combos)}] mean RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f}) | {config}"
        )

    best_model = train_pca_poly_regressor(X, y, **best_config)

    return {
        "best_config": best_config,
        "best_model": best_model,
        "cv_results": cv_results,
    }
