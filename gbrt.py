from itertools import product
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, StratifiedKFold


ClassifierParams = Dict[str, Any]
RegressorParams = Dict[str, Any]


def _apply_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    params = defaults.copy()
    params.update(overrides)
    return params


def _needs_multi_output(y: np.ndarray) -> bool:
    return y.ndim > 1 and y.shape[1] > 1


def _expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def train_gbrt_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **overrides: Any,
) -> GradientBoostingClassifier:
    defaults: ClassifierParams = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }
    params = _apply_defaults(defaults, overrides)
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_gbrt_classifier(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_mse": mean_squared_error(y_test, y_pred_test),
    }


def grid_search_gbrt_classifier(
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
) -> Dict[str, Any]:
    combos = _expand_param_grid(param_grid)
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_score = -np.inf
    best_config: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for config in combos:
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            model = train_gbrt_classifier(X[train_idx], y[train_idx], **config)
            val_pred = model.predict(X[val_idx])
            scores.append(accuracy_score(y[val_idx], val_pred))

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        result_entry = config.copy()
        result_entry.update(
            {"mean_val_accuracy": mean_score, "std_val_accuracy": std_score}
        )
        results.append(result_entry)

        if mean_score > best_score:
            best_score = mean_score
            best_config = config

    best_model = train_gbrt_classifier(X, y, **best_config)

    return {
        "best_config": best_config,
        "best_model": best_model,
        "cv_results": results,
    }


def train_gbrt_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **overrides: Any,
):
    defaults: RegressorParams = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    }
    params = _apply_defaults(defaults, overrides)
    base_model = GradientBoostingRegressor(**params)
    model = MultiOutputRegressor(base_model) if _needs_multi_output(y_train) else base_model
    model.fit(X_train, y_train)
    return model


def evaluate_gbrt_regressor(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mee = np.linalg.norm(
        (y_pred_train - y_train).reshape(len(y_train), -1), axis=1
    ).mean()
    test_mee = np.linalg.norm(
        (y_pred_test - y_test).reshape(len(y_test), -1), axis=1
    ).mean()

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_rmse": np.sqrt(test_mse),
        "train_mee": train_mee,
        "test_mee": test_mee,
    }


def grid_search_gbrt_regressor(
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
) -> Dict[str, Any]:
    combos = _expand_param_grid(param_grid)
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_score = np.inf
    best_config: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for [i, config] in enumerate(combos):
        mees = []
        for train_idx, val_idx in cv.split(X):
            model = train_gbrt_regressor(X[train_idx], y[train_idx], **config)
            val_pred = model.predict(X[val_idx])
            val_err = (val_pred - y[val_idx]).reshape(len(val_idx), -1)
            val_mee = np.linalg.norm(val_err, axis=1).mean()
            mees.append(val_mee)

        mean_mee = float(np.mean(mees))
        std_mee = float(np.std(mees))
        result_entry = config.copy()
        result_entry.update({"mean_val_mee": mean_mee, "std_val_mee": std_mee})
        results.append(result_entry)

        if mean_mee < best_score:
            best_score = mean_mee
            best_config = config

        print(f"Config {i + 1}: Avg MEE: {mean_mee:.4f} (+/- {std_mee:.4f}) | {config}")

    best_model = train_gbrt_regressor(X, y, **best_config)

    return {
        "best_config": best_config,
        "best_model": best_model,
        "cv_results": results,
    }
