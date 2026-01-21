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
    y_evolution = df[["y_evolution_1", "y_evolution_2", "y_evolution_3", "y_evolution_4"]].values
    y_snapped = df[
        ["y_test5_snapped_1", "y_test5_snapped_2", "y_test5_snapped_3", "y_test5_snapped_4"]
    ].values
    z_snapped = df["z_test5_snapped"].values
    z_forest = df["z_forest"].values

    y_err = np.linalg.norm(y_evolution - y_true, axis=1)
    d_perp_vals = np.array([_d_perp(y_hat) for y_hat in y_evolution])
    z_diff = np.abs(z_snapped - z_forest)
    use_evolution = (z_diff > 0.2) & (d_perp_vals < 0.2)
    multiplier = 0.1
    y_ens = np.where(use_evolution[:, None], multiplier * y_evolution + (1 - multiplier) * y_snapped, y_snapped)
    y_ens_err = np.linalg.norm(y_ens - y_true, axis=1)

    out_dir = Path("plots/9")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "y_evolution_error_vs_d_perp.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_err, d_perp_vals, s=18, alpha=0.7, color="tab:blue")
    ax.set_xlabel("Error |y_evolution - y_true|")
    ax.set_ylabel("d_perp(y_evolution)")
    ax.set_title("y_evolution error vs d_perp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "y_evolution_err_vs_y_snapped_err.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    err_evolution = np.linalg.norm(y_evolution - y_true, axis=1)
    err_snapped = np.linalg.norm(y_snapped - y_true, axis=1)
    ax.scatter(err_evolution, err_snapped, s=18, alpha=0.7, color="tab:green")
    lim = max(max(err_evolution), max(err_snapped))
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Error |y_evolution - y_true|")
    ax.set_ylabel("Error |y(z_snapped) - y_true|")
    ax.set_title("y_evolution error vs y(z_snapped) error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    order = np.argsort(d_perp_vals)
    d_sorted = d_perp_vals[order]
    err_sorted = y_err[order]
    window = 50
    if len(err_sorted) >= window:
        kernel = np.ones(window) / window
        avg_err = np.convolve(err_sorted, kernel, mode="valid")
        d_centers = d_sorted[window - 1:]

        out_path = out_dir / "alpha_vs_mee_d_perp_sliding.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(d_centers, avg_err, color="tab:blue", linewidth=2, label="MEE (window)")
        ax.set_xlabel("alpha (d_perp threshold)")
        ax.set_ylabel("MEE on points within sliding window")
        ax.set_title("MEE vs d_perp (sliding window)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

    cum_err = np.cumsum(err_sorted)
    counts = np.arange(1, len(err_sorted) + 1)
    mee_by_alpha = cum_err / counts
    out_path = out_dir / "alpha_vs_mee_d_perp_prefix.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(d_sorted, mee_by_alpha, color="tab:blue", linewidth=2, label="MEE (prefix)")
    ax.set_xlabel("alpha (d_perp threshold)")
    ax.set_ylabel("MEE on points with d_perp < alpha")
    ax2 = ax.twinx()
    ax2.semilogx(d_sorted, counts, color="tab:orange", linewidth=2, label="Count")
    ax2.set_ylabel("Number of points with d_perp < alpha")
    ax.set_title("MEE vs d_perp (prefix)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = out_dir / "z_snapped_minus_z_forest_vs_y_snapped_err.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    y_snapped_err = np.linalg.norm(y_snapped - y_true, axis=1)
    ax.scatter(z_diff, y_snapped_err, s=18, alpha=0.7, color="tab:purple")
    ax.set_xlabel("|z_snapped - z_forest|")
    ax.set_ylabel("|y(z_snapped) - y_true|")
    ax.set_title("z_snapped - z_forest vs y(z_snapped) error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    order = np.argsort(z_diff)
    z_sorted = z_diff[order]
    err_sorted = y_snapped_err[order]
    window = 50
    if len(err_sorted) >= window:
        kernel = np.ones(window) / window
        avg_err = np.convolve(err_sorted, kernel, mode="valid")
        z_centers = z_sorted[window - 1:]

        out_path = out_dir / "z_diff_vs_y_snapped_err_sliding_avg.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(z_centers, avg_err, color="tab:blue", linewidth=2)
        ax.set_xlabel("|z_snapped - z_forest| (sorted)")
        ax.set_ylabel("Sliding average |y(z_snapped) - y_true| (window=20)")
        ax.set_title("Sliding average y(z_snapped) error vs |z_snapped - z_forest|")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

    cum_err = np.cumsum(err_sorted)
    counts = np.arange(1, len(err_sorted) + 1)
    mee_by_alpha = cum_err / counts
    out_path = out_dir / "z_diff_vs_y_snapped_err_prefix.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(z_sorted, mee_by_alpha, color="tab:blue", linewidth=2, label="MEE (prefix)")
    ax.set_xlabel("|z_snapped - z_forest| (sorted)")
    ax.set_ylabel("MEE on points with |z_snapped - z_forest| < alpha")
    ax2 = ax.twinx()
    ax2.semilogx(z_sorted, counts, color="tab:orange", linewidth=2, label="Count")
    ax2.set_ylabel("Number of points with |z_snapped - z_forest| < alpha")
    ax.set_title("Prefix MEE vs |z_snapped - z_forest|")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    order = np.argsort(z_diff)
    z_sorted = z_diff[order]
    err_sorted = y_ens_err[order]
    if len(err_sorted) >= window:
        kernel = np.ones(window) / window
        avg_err = np.convolve(err_sorted, kernel, mode="valid")
        z_centers = z_sorted[window - 1:]

        out_path = out_dir / "z_diff_vs_y_ens_err_sliding_avg.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(z_centers, avg_err, color="tab:blue", linewidth=2)
        ax.set_xlabel("|z_snapped - z_forest| (sorted)")
        ax.set_ylabel("Sliding average |y_ens - y_true| (window=20)")
        ax.set_title("Sliding average y_ens error vs |z_snapped - z_forest|")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

    cum_err = np.cumsum(err_sorted)
    counts = np.arange(1, len(err_sorted) + 1)
    mee_by_alpha = cum_err / counts
    out_path = out_dir / "z_diff_vs_y_ens_err_prefix.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(z_sorted, mee_by_alpha, color="tab:blue", linewidth=2, label="MEE (prefix)")
    ax.set_xlabel("|z_snapped - z_forest| (sorted)")
    ax.set_ylabel("MEE on points with |z_snapped - z_forest| < alpha")
    ax2 = ax.twinx()
    ax2.semilogx(z_sorted, counts, color="tab:orange", linewidth=2, label="Count")
    ax2.set_ylabel("Number of points with |z_snapped - z_forest| < alpha")
    ax.set_title("Prefix MEE vs |z_snapped - z_forest| (y_ens)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
