import numpy as np
from pathlib import Path

from utils import load_cup_data


def compute_alpha_with_beta(y_abs: np.ndarray, x_abs: np.ndarray, beta: float, eps: float = 1e-12) -> float:
    """Given beta >= 0, return the smallest alpha so that |y| <= alpha*|x| + beta for all samples."""
    residual = np.maximum(y_abs - beta, 0.0)
    ratio = residual / (x_abs + eps)
    return float(ratio.max())


def main():
    file_path = Path("data/CUP/ML-CUP25-TR.csv")
    X, y = load_cup_data(file_path)
    y_abs = np.abs(y[:, 0])  # |y1|
    eps = 1e-12

    # Choose a small grid of beta values (percentiles of |y1|)
    beta_vals = np.percentile(y_abs, [0, 25, 50, 75, 90, 95])

    print(f"Beta grid (percentiles of |y1|): {np.round(beta_vals, 6)}\n")

    for i in range(X.shape[1]):
        xi_abs = np.abs(X[:, i])
        ratios = np.abs(y_abs) / (xi_abs + eps)
        alpha_no_beta = ratios.max()
        print(f"x{i+1}: alpha (beta=0) = {alpha_no_beta:.6f}")
        for beta in beta_vals[1:]:  # skip 0 since reported above
            alpha_beta = compute_alpha_with_beta(y_abs, xi_abs, beta, eps)
            print(f"  beta={beta:.4f} -> alpha={alpha_beta:.6f}")
        print()


if __name__ == "__main__":
    main()
