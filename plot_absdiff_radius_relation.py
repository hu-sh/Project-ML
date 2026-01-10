import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_inputs(file_path: Path) -> np.ndarray:
    df = pd.read_csv(file_path, comment="#", header=None)
    # Drop ID column and keep first 12 input columns
    return df.iloc[:, 1:13].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot absdiff(x11, x12) against a scaled radius(x11, x12) relation."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/CUP/ML-CUP25-TR.csv"),
        help="Path to CUP training CSV (default: data/CUP/ML-CUP25-TR.csv)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.4081,
        help="Scale applied to radius_x_11_x_12 (default: 1.4081)",
    )
    args = parser.parse_args()

    X = load_inputs(args.file)
    x11 = X[:, 10]
    x12 = X[:, 11]
    absdiff = np.abs(x11 - x12)
    radius = np.hypot(x11, x12)
    scaled_radius = args.scale * radius
    residual = absdiff - scaled_radius

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.scatter(scaled_radius, absdiff, alpha=0.4, s=12, color="tab:blue")
    lim = max(np.max(np.abs(scaled_radius)), np.max(np.abs(absdiff))) * 1.05
    ax0.plot([0, lim], [0, lim], "--", color="gray", alpha=0.7, label="y = x")
    ax0.set_xlim(0, lim)
    ax0.set_ylim(0, lim)
    ax0.set_xlabel(f"{args.scale:.4f} * radius_x_11_x_12")
    ax0.set_ylabel("absdiff_x_11_x_12")
    ax0.set_title("absdiff vs scaled radius")
    ax0.legend()
    ax0.grid(True, linestyle=":", alpha=0.5)

    ax1.hist(residual, bins=50, color="tab:orange", alpha=0.8)
    ax1.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax1.set_title("Residual: absdiff - scaled_radius")
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle=":", alpha=0.5)

    mean_res = residual.mean()
    median_res = np.median(residual)
    ax1.text(
        0.95,
        0.95,
        f"mean={mean_res:.4f}\nmedian={median_res:.4f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
