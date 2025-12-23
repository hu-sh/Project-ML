import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_cup(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path, comment="#", header=None)
    vals = df.iloc[:, 1:]
    X = vals.iloc[:, :12].to_numpy()
    Y = vals.iloc[:, 12:].to_numpy()
    return X, Y


def best_scale(target: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.dot(candidate, candidate))
    if denom == 0:
        return 0.0
    return float(np.dot(target, candidate) / denom)


def normalized_stats(target: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    residual = np.abs(target - pred)
    denom = np.maximum(np.maximum(np.abs(target), np.abs(pred)), 1e-6)
    res_norm = residual / denom
    return float(res_norm.mean()), float(np.median(res_norm)), float(res_norm.max())


def rank_scaled_radii(X: np.ndarray, target_radius: np.ndarray) -> List[dict]:
    pairs = list(combinations(range(X.shape[1]), 2))
    ranked = []
    for i, j in pairs:
        x_rad = np.hypot(X[:, i], X[:, j])
        scale = best_scale(target_radius, x_rad)
        pred = scale * x_rad
        mean_norm, median_norm, max_norm = normalized_stats(target_radius, pred)
        ranked.append(
            {
                "pair": (i + 1, j + 1),
                "scale": scale,
                "mean_norm": mean_norm,
                "median_norm": median_norm,
                "max_norm": max_norm,
            }
        )
    ranked.sort(key=lambda r: r["mean_norm"])
    return ranked


def fit_linear_combo(target_radius: np.ndarray, X: np.ndarray, ranked: List[dict], top_k: int):
    chosen = ranked[:top_k]
    cols = []
    labels = []
    for item in chosen:
        i, j = item["pair"]
        x_rad = np.hypot(X[:, i - 1], X[:, j - 1])
        cols.append(item["scale"] * x_rad)
        labels.append(item)
    if not cols:
        return [], None
    Xmat = np.column_stack(cols)
    w, _, _, _ = np.linalg.lstsq(Xmat, target_radius, rcond=None)
    pred = Xmat.dot(w)
    stats = normalized_stats(target_radius, pred)
    return list(zip(labels, w)), stats


def main():
    parser = argparse.ArgumentParser(
        description="Find scaled input radii that approximate a target radius, and fit a linear combo of the best."
    )
    parser.add_argument(
        "--file", type=Path, default=Path("data/CUP/ML-CUP25-TR.csv"), help="CUP training CSV (default: data/CUP/ML-CUP25-TR.csv)"
    )
    parser.add_argument("--target-i", type=int, default=3, help="Target column index (1-based) for radius component 1 (default: 3)")
    parser.add_argument("--target-j", type=int, default=4, help="Target column index (1-based) for radius component 2 (default: 4)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top relations to use in the linear combo (default: 10)")
    parser.add_argument("--show-top", type=int, default=5, help="Number of top single relations to print (default: 5)")
    args = parser.parse_args()

    X, Y = load_cup(args.file)
    ti = args.target_i - 1
    tj = args.target_j - 1
    target_radius = np.hypot(Y[:, ti], Y[:, tj])

    ranked = rank_scaled_radii(X, target_radius)

    print(f"Top {min(args.show_top, len(ranked))} single scaled radii (by mean normalized residual):")
    for item in ranked[: args.show_top]:
        i, j = item["pair"]
        print(
            f"radius_target_{args.target_i}_{args.target_j} ~ {item['scale']:.4f} * radius_input_{i}_{j} | "
            f"mean_norm={item['mean_norm']:.4f}, median_norm={item['median_norm']:.4f}, max_norm={item['max_norm']:.4f}"
        )

    combo, stats = fit_linear_combo(target_radius, X, ranked, args.top_k)
    if combo and stats:
        mean_norm, median_norm, max_norm = stats
        print(f"\nLinear combo using top {len(combo)} scaled radii:")
        for (item, w) in combo:
            i, j = item["pair"]
            print(f" w={w:.4f} * ({item['scale']:.4f} * radius_input_{i}_{j})")
        print(
            f"combined residuals: mean_norm={mean_norm:.4f}, median_norm={median_norm:.4f}, max_norm={max_norm:.4f}"
        )
    else:
        print("No features available for linear combination.")


if __name__ == "__main__":
    main()
