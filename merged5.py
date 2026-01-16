from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


PRED_CSV = Path("plots/errors/predictions_cv.csv")


def _calculate_mee(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {PRED_CSV}: {missing}")


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)
    x_cols = [f"x_{i}" for i in range(1, 13)]
    pred_cols = ["z_pred", "z_est_raw", "z_pred_pca", "y_pred_1", "y_pred_2", "y_pred_3", "y_pred_4"]
    y_cols = ["y_val_1", "y_val_2", "y_val_3", "y_val_4"]
    _require_columns(df, ["fold"] + x_cols + pred_cols + y_cols)

    X_feat = df[x_cols + pred_cols].values
    Y = df[y_cols].values
    folds = df["fold"].values.astype(int)
    unique_folds = np.unique(folds)
    y_pred_all = np.zeros_like(Y)
    for fold in unique_folds:
        train_mask = folds != fold
        val_mask = folds == fold

        X_train = X_feat[train_mask]
        Y_train = Y[train_mask]
        X_val = X_feat[val_mask]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
        mlp.fit(X_train_sc, Y_train)
        y_pred_all[val_mask] = mlp.predict(X_val_sc)

        mee_fold = _calculate_mee(Y[val_mask], y_pred_all[val_mask])
        print(f"Fold {fold} MEE MLP (augmented features): {mee_fold:.6f}", flush=True)

    mee = _calculate_mee(Y, y_pred_all)
    print(f"Overall MEE MLP (augmented features): {mee:.6f}", flush=True)


if __name__ == "__main__":
    main()
