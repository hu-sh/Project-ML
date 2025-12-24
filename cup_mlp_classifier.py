import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from utils import load_cup_data
from mlp import train_model, device


def solve_magnitudes(sum_abs: float, r: float) -> Tuple[float, float]:
    """Given s = |y1|+|y2| and r = sqrt(y1^2 + y2^2), return the two candidate magnitudes."""
    disc = 2 * r * r - sum_abs * sum_abs
    disc = max(disc, 0.0)
    delta = np.sqrt(disc)
    a = (sum_abs + delta) / 2.0
    b = (sum_abs - delta) / 2.0
    return a, b


def build_classifier_labels(y: np.ndarray) -> np.ndarray:
    """Encode permutation + signs into a single class label (8 classes)."""
    y1, y2 = y[:, 0], y[:, 1]
    a1, a2 = np.abs(y1), np.abs(y2)
    perm_flag = (a1 >= a2).astype(int)  # 1 if |y1| >= |y2|
    sign1 = (y1 >= 0).astype(int)
    sign2 = (y2 >= 0).astype(int)
    return (perm_flag << 2) | (sign1 << 1) | sign2  # bits [perm, sign1, sign2]


def decode_class(cls: int) -> Tuple[int, int, int]:
    perm_flag = (cls >> 2) & 1
    sign1 = (cls >> 1) & 1
    sign2 = cls & 1
    return perm_flag, sign1, sign2


def main():
    train_file = Path("data/CUP/ML-CUP25-TR.csv")
    print(f"--- Stacking MLP + classifier on {train_file} ---")

    X_raw, y_raw = load_cup_data(train_file)
    y_main = y_raw[:, :4]

    # Aux targets
    y_r12 = np.hypot(y_raw[:, 0], y_raw[:, 1]).reshape(-1, 1)
    y_diff34 = (y_raw[:, 2] - y_raw[:, 3]).reshape(-1, 1)
    y_meanabs = np.mean(np.abs(y_main), axis=1, keepdims=True)
    y_aux = np.hstack([y_r12, y_diff34, y_meanabs])  # shape (n,3)

    # Inputs: signed, excluding x6 (as in cup_mlp.py)
    input_indices_raw = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    X_signed = X_raw[:, input_indices_raw]

    # Aux model split
    X_train_aux, X_val_aux, y_train_aux, y_val_aux = train_test_split(
        X_signed, y_aux, test_size=0.2, random_state=42
    )

    scaler_x_aux = StandardScaler()
    X_train_aux = scaler_x_aux.fit_transform(X_train_aux)
    X_val_aux = scaler_x_aux.transform(X_val_aux)

    scaler_y_aux = StandardScaler()
    y_train_aux = scaler_y_aux.fit_transform(y_train_aux)
    y_val_aux = scaler_y_aux.transform(y_val_aux)

    X_train_aux_t = torch.FloatTensor(X_train_aux)
    y_train_aux_t = torch.FloatTensor(y_train_aux)
    X_val_aux_t = torch.FloatTensor(X_val_aux)
    y_val_aux_t = torch.FloatTensor(y_val_aux)

    # MLP config (reuse cup_mlp.py defaults)
    config = {
        "hidden_layers": [64, 32],
        "activation": "ReLU",
        "lr": 0.0005,
        "weight_decay": 1e-5,
        "momentum": 0.9,
        "epochs": 800,
        "batch_size": 32,
        "optim": "sgd",
        "use_scheduler": True,
        "loss": "MSE",
        "es": True,
        "patience": 40,
    }

    print("\nTraining auxiliary MLP (r12, y3-y4, mean|y|)...")
    aux_model, aux_history, aux_stop = train_model(
        config,
        input_dim=X_train_aux.shape[1],
        X_train=X_train_aux_t,
        y_train=y_train_aux_t,
        X_val=X_val_aux_t,
        y_val=y_val_aux_t,
        task_type="regression",
    )
    print(f"Aux training stopped at epoch {aux_stop}")

    # Aux predictions on full dataset
    X_all_aux = scaler_x_aux.transform(X_signed)
    X_all_aux_t = torch.FloatTensor(X_all_aux).to(device)
    aux_model.eval()
    with torch.no_grad():
        aux_pred_all_scaled = aux_model(X_all_aux_t).cpu().numpy()
    aux_pred_all = scaler_y_aux.inverse_transform(aux_pred_all_scaled)

    # Aux MAE on validation split (no inverse scaling)
    with torch.no_grad():
        aux_pred_val_scaled = aux_model(X_val_aux_t.to(device)).cpu().numpy()
    aux_true_val_scaled = y_val_aux_t.numpy()
    aux_mae_scaled = np.mean(np.abs(aux_pred_val_scaled - aux_true_val_scaled), axis=0)
    print("Aux MAE (scaled) [r12, y3-y4, mean|y|]:", np.round(aux_mae_scaled, 4))

    # Labels for permutation/sign (8 classes) computed from ground truth
    cls_labels = build_classifier_labels(y_main)

    # Split for classifier/main reconstruction
    (
        X_train_cls,
        X_val_cls,
        y_train_main,
        y_val_main,
        aux_train,
        aux_val,
        cls_train,
        cls_val,
    ) = train_test_split(
        X_signed, y_main, aux_pred_all, cls_labels, test_size=0.2, random_state=42, stratify=cls_labels
    )

    # Build classifier features (optionally scale)
    scaler_x_cls = StandardScaler()
    cls_X_train = scaler_x_cls.fit_transform(np.hstack([X_train_cls, aux_train]))
    cls_X_val = scaler_x_cls.transform(np.hstack([X_val_cls, aux_val]))

    cls_clf = RandomForestClassifier(n_estimators=400, random_state=42)
    cls_clf.fit(cls_X_train, cls_train)

    # Evaluate classifier on validation split
    cls_pred_val = cls_clf.predict(cls_X_val)
    perm_true = [(c >> 2) & 1 for c in cls_val]
    perm_pred = [(c >> 2) & 1 for c in cls_pred_val]
    sign1_true = [(c >> 1) & 1 for c in cls_val]
    sign1_pred = [(c >> 1) & 1 for c in cls_pred_val]
    sign2_true = [c & 1 for c in cls_val]
    sign2_pred = [c & 1 for c in cls_pred_val]
    cls_acc = (cls_pred_val == cls_val).mean()
    perm_acc = np.mean(np.array(perm_true) == np.array(perm_pred))
    sign1_acc = np.mean(np.array(sign1_true) == np.array(sign1_pred))
    sign2_acc = np.mean(np.array(sign2_true) == np.array(sign2_pred))
    print(f"Classifier accuracy: total={cls_acc:.4f}, perm={perm_acc:.4f}, sign1={sign1_acc:.4f}, sign2={sign2_acc:.4f}")

    # Reconstruct y on validation split
    r12_from_diff_val = 0.5463 * np.abs(aux_val[:, 1:2])
    preds = []
    for X_row, aux_row, r12_rel, cls_pred in zip(X_val_cls, aux_val, r12_from_diff_val, cls_clf.predict(cls_X_val)):
        r12_pred = float(aux_row[0])
        diff34_pred = float(aux_row[1])
        meanabs_pred = float(aux_row[2])
        r12_rel_scalar = float(np.squeeze(r12_rel))
        # r12_used = 0.5 * r12_pred + 0.5 * r12_rel_scalar
        r12_used = r12_pred

        s3 = 1.0 if diff34_pred >= 0 else -1.0
        s4 = -s3
        absdiff34 = abs(diff34_pred)
        y3 = s3 * (absdiff34 / 2.0)
        y4 = s4 * (absdiff34 / 2.0)

        sum_abs_y12 = 4 * meanabs_pred - abs(y3) - abs(y4)
        a, b = solve_magnitudes(sum_abs_y12, r12_used)

        perm_flag, sign1_bit, sign2_bit = decode_class(cls_pred)
        sign1 = 1.0 if sign1_bit == 1 else -1.0
        sign2 = 1.0 if sign2_bit == 1 else -1.0
        if perm_flag == 1:
            abs1, abs2 = a, b
        else:
            abs1, abs2 = b, a
        y1 = sign1 * abs1
        y2 = sign2 * abs2
        preds.append([y1, y2, y3, y4])

    preds = np.array(preds)
    mae = np.mean(np.abs(preds - y_val_main), axis=0)
    rmse = np.sqrt(np.mean((preds - y_val_main) ** 2))
    mee_norm = np.linalg.norm(preds - y_val_main, axis=1).mean()

    print("MAE per output (y1..y4) stacking MLP reconstruction):", np.round(mae, 4))
    print(f"RMSE complessivo (y1..y4): {rmse:.4f}")
    print(f"MEE: {mee_norm:.4f}")


if __name__ == "__main__":
    main()
