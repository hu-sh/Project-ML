from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path("plots/9/predictions_partial_final.csv")


def _y_of_z(z):
    z = np.asarray(z)
    k1 = 0.5463
    k2 = 1.1395
    y1 = k1 * z * np.cos(k2 * z)
    y2 = k1 * z * np.sin(k2 * z)
    sum_y34 = -z * np.cos(2 * z)
    y3 = (sum_y34 + z) / 2.0
    y4 = (sum_y34 - z) / 2.0
    return np.column_stack([y1, y2, y3, y4])


def _d_perp(y_hat, R=10.0, n_grid=10001):
    y_hat = np.asarray(y_hat)
    z0 = y_hat[2] - y_hat[3]
    zs = np.linspace(z0 - R, z0 + R, n_grid)
    ys = _y_of_z(zs)
    dists = np.linalg.norm(ys - y_hat, axis=1)
    return float(np.min(dists))


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    y_true = df[["y_true_1", "y_true_2", "y_true_3", "y_true_4"]].values
    y_forest = df[["y_forest_1", "y_forest_2", "y_forest_3", "y_forest_4"]].values

    y_err = np.linalg.norm(y_forest - y_true, axis=1)
    d_perp_vals = np.array([_d_perp(y_hat) for y_hat in y_forest])

    out_dir = Path("plots/9")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "y_forest_error_vs_d_perp.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_err, d_perp_vals, s=18, alpha=0.7, color="tab:blue")
    ax.set_xlabel("Error |y_forest - y_true|")
    ax.set_ylabel("d_perp(y_forest)")
    ax.set_title("y_forest error vs d_perp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
