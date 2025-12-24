import numpy as np
from pathlib import Path

from utils import load_cup_data


def compute_alphas(y_abs: np.ndarray, X_abs: np.ndarray, betas: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute alpha_i(beta) such that |y| <= alpha_i * |x_i| + beta for all samples.
    Returns array shape (len(betas), n_features).
    """
    alphas = []
    for beta in betas:
        residual = np.maximum(y_abs - beta, 0.0)
        # ratio shape (n, d)
        ratios = residual[:, None] / (X_abs + eps)
        alphas.append(ratios.max(axis=0))
    return np.vstack(alphas)


def main():
    file_path = Path("data/CUP/ML-CUP25-TR.csv")
    X, y = load_cup_data(file_path)
    y1_abs = np.abs(y[:, 0])
    X_abs = np.abs(X)

    betas = np.percentile(y1_abs, [0, 25, 50, 75, 90, 95])
    alphas = compute_alphas(y1_abs, X_abs, betas)

    print("Betas:", np.round(betas, 4))
    print("Alphas shape:", alphas.shape)

    # For each beta, compute per-sample upper bound: min_i alpha_i(beta)*|x_i| + beta
    for idx, beta in enumerate(betas):
        alpha_beta = alphas[idx]  # shape (d,)
        ub = (alpha_beta[None, :] * X_abs + beta).min(axis=1)
        gap = ub - y1_abs
        print(f"\nBeta={beta:.4f}")
        print(
            f"Upper bound stats: mean ub={ub.mean():.4f}, median ub={np.median(ub):.4f}, "
            f"mean gap={gap.mean():.4f}, median gap={np.median(gap):.4f}, "
            f"pct within 0.1 of y1={((gap<=0.1*y1_abs).mean()):.3f}"
        )

    # Combine all betas: take the minimum bound across beta choices (all inequalities hold)
    ub_all = np.minimum.reduce(
        [(alphas[idx][None, :] * X_abs + betas[idx]).min(axis=1) for idx in range(len(betas))]
    )
    gap_all = ub_all - y1_abs
    print(
        "\nCombined (min over betas) bound stats:"
        f" mean ub={ub_all.mean():.4f}, median ub={np.median(ub_all):.4f}, "
        f"mean gap={gap_all.mean():.4f}, median gap={np.median(gap_all):.4f}, "
        f"pct within 0.1 of y1={((gap_all<=0.1*y1_abs).mean()):.3f}"
    )


if __name__ == "__main__":
    main()
