import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_radii(file_path: Path, n_inputs: int = 12) -> np.ndarray:
    """Load TARGET_1 and TARGET_2 radii from the CUP training CSV."""
    df = pd.read_csv(file_path, comment="#", header=None)
    # Drop the ID column, then slice the first two target columns.
    targets = df.iloc[:, 1 + n_inputs : 1 + n_inputs + 2].to_numpy()
    return np.hypot(targets[:, 0], targets[:, 1])


def sse_for_alpha(r: np.ndarray, alpha: float) -> float:
    nearest = alpha * np.round(r / alpha)
    err = r - nearest
    return float(np.sum(err**2))


def best_alpha(
    r: np.ndarray, low: float, high: float, coarse_steps: int, refine_window: float, refine_steps: int
) -> Tuple[float, float]:
    alphas = np.linspace(low, high, coarse_steps)
    sse = np.array([sse_for_alpha(r, a) for a in alphas])
    best_idx = int(sse.argmin())
    coarse_alpha = float(alphas[best_idx])

    refine_low = coarse_alpha - refine_window
    refine_high = coarse_alpha + refine_window
    alphas_ref = np.linspace(refine_low, refine_high, refine_steps)
    sse_ref = np.array([sse_for_alpha(r, a) for a in alphas_ref])
    best_idx_ref = int(sse_ref.argmin())
    return float(alphas_ref[best_idx_ref]), float(sse_ref[best_idx_ref]), alphas, sse, alphas_ref, sse_ref


def plot_sse(alphas, sse, alphas_ref, sse_ref, best_alpha_val: float, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(alphas, sse, label="coarse SSE")
    axes[0].axvline(best_alpha_val, color="red", linestyle="--", label=f"best α≈{best_alpha_val:.4f}")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("SSE")
    axes[0].set_title("Coarse search")
    axes[0].legend()

    axes[1].plot(alphas_ref, sse_ref, label="refined SSE")
    axes[1].axvline(best_alpha_val, color="red", linestyle="--", label=f"best α≈{best_alpha_val:.4f}")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("SSE")
    axes[1].set_title("Refinement")
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / "alpha_sse.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate alpha for concentric targets.")
    parser.add_argument(
        "--file",
        default="data/CUP/ML-CUP25-TR.csv",
        type=Path,
        help="Path to the CUP training CSV (default: data/CUP/ML-CUP25-TR.csv)",
    )
    parser.add_argument("--low", type=float, default=2.0, help="Lower bound for alpha search (default: 2.0)")
    parser.add_argument("--high", type=float, default=4.0, help="Upper bound for alpha search (default: 4.0)")
    parser.add_argument(
        "--coarse-steps", type=int, default=2001, help="Number of coarse grid steps across [low, high] (default: 2001)"
    )
    parser.add_argument(
        "--refine-window",
        type=float,
        default=0.05,
        help="Half-width around coarse optimum for refinement (default: 0.05)",
    )
    parser.add_argument(
        "--refine-steps", type=int, default=2001, help="Number of refinement steps (default: 2001)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save SSE vs alpha plots to ./plots/alpha_sse.png"
    )
    args = parser.parse_args()

    radii = load_radii(args.file)
    alpha, sse, alphas, sse_all, alphas_ref, sse_ref = best_alpha(
        radii,
        low=args.low,
        high=args.high,
        coarse_steps=args.coarse_steps,
        refine_window=args.refine_window,
        refine_steps=args.refine_steps,
    )

    nearest = alpha * np.round(radii / alpha)
    err = np.abs(radii - nearest)

    print(f"Best alpha: {alpha:.6f}")
    print(f"Minimum SSE: {sse:.6f}")
    print(f"Mean |err|: {err.mean():.6f}")
    print(f"Std  |err|: {err.std():.6f}")
    print(f"Median |err|: {np.median(err):.6f}")
    print(f"Max |err|: {err.max():.6f}")
    if args.plot:
        out_path = plot_sse(alphas, sse_all, alphas_ref, sse_ref, alpha, Path("plots"))
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
