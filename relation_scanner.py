import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


Pair = Tuple[int, int]
FeatureDict = Dict[str, np.ndarray]


def load_cup(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path, comment="#", header=None)
    # Drop ID column
    values = df.iloc[:, 1:]
    inputs = values.iloc[:, :12].to_numpy()
    targets = values.iloc[:, 12:].to_numpy()
    return inputs, targets


def pair_names(n_cols: int) -> List[Pair]:
    return [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]


def build_features(arr: np.ndarray, prefix: str) -> FeatureDict:
    feats: FeatureDict = {}
    # Single-variable features
    for i in range(arr.shape[1]):
        feats[f"raw_{prefix}_{i+1}"] = arr[:, i]
        feats[f"abs_{prefix}_{i+1}"] = np.abs(arr[:, i])
        feats[f"sign_{prefix}_{i+1}"] = np.sign(arr[:, i])
    # Aggregates across all variables
    eps = 1e-12
    geom_all = np.exp(np.mean(np.log(np.abs(arr) + eps), axis=1))
    mean_abs_all = np.mean(np.abs(arr), axis=1)
    feats[f"geom_all_{prefix}"] = geom_all
    feats[f"meanabs_all_{prefix}"] = mean_abs_all
    # Aggregates excluding x_6 (only for inputs)
    if prefix == "input" and arr.shape[1] >= 6:
        mask = np.ones(arr.shape[1], dtype=bool)
        mask[5] = False  # zero-based index for x_6
        arr_excl6 = arr[:, mask]
        geom_excl6 = np.exp(np.mean(np.log(np.abs(arr_excl6) + eps), axis=1))
        mean_abs_excl6 = np.mean(np.abs(arr_excl6), axis=1)
        feats[f"geom_all_{prefix}_excl6"] = geom_excl6
        feats[f"meanabs_all_{prefix}_excl6"] = mean_abs_excl6

    pairs = pair_names(arr.shape[1])
    for i, j in pairs:
        pfx = f"{prefix}_{i+1}_{j+1}"
        a_i, a_j = arr[:, i], arr[:, j]
        feats[f"prod_{pfx}"] = a_i * a_j
        feats[f"sum_{pfx}"] = a_i + a_j
        feats[f"diff_{pfx}"] = a_i - a_j
        feats[f"radius_{pfx}"] = np.hypot(a_i, a_j)
        feats[f"angle_{pfx}"] = np.arctan2(a_j, a_i)
        feats[f"geom_{pfx}"] = np.sqrt(abs(a_i * a_j))
    return feats


def angle_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = np.abs(a - b)
    # Wrap to [0, pi]
    return np.minimum(raw, 2 * math.pi - raw)


def compute_near_equalities(
    inputs: FeatureDict,
    targets: FeatureDict,
    thresholds: Tuple[float, float],
    top_n: int,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    small_thr, medium_thr = thresholds

    for name_t, vals_t in targets.items():
        base_name = name_t.split("_", 1)[0]
        # Compare only features with the same operation type (prod, sum, diff, radius, angle)
        for name_x, vals_x in inputs.items():
            if not name_x.startswith(base_name):
                continue
            if base_name == "angle":
                residual = angle_residual(vals_t, vals_x)
                candidates = [("1.0", residual)]
            else:
                residual_unscaled = np.abs(vals_t - vals_x)
                # Fit best scalar multiplier a = argmin ||t - a*x||^2
                denom = float(np.dot(vals_x, vals_x))
                if denom > 0:
                    a = float(np.dot(vals_t, vals_x) / denom)
                    residual_scaled = np.abs(vals_t - a * vals_x)
                    candidates = [("1.0", residual_unscaled), (f"{a:.4f}", residual_scaled)]
                else:
                    candidates = [("1.0", residual_unscaled)]

            for scale, residual in candidates:
                denom = np.maximum(np.maximum(np.abs(vals_t), np.abs(vals_x)), 1e-6)
                residual_norm = residual / denom

                mean_abs = float(residual.mean())
                median_abs = float(np.median(residual))
                max_abs = float(residual.max())
                mean_abs_norm = float(residual_norm.mean())
                median_abs_norm = float(np.median(residual_norm))
                max_abs_norm = float(residual_norm.max())
                pct_small = float((residual_norm <= small_thr).mean())
                pct_med = float((residual_norm <= medium_thr).mean())
                records.append(
                    {
                        "relation": f"{name_t} ~ {scale} * {name_x}",
                        "mean_abs": mean_abs,
                        "median_abs": median_abs,
                        "max_abs": max_abs,
                        "mean_abs_norm": mean_abs_norm,
                        "median_abs_norm": median_abs_norm,
                        "max_abs_norm": max_abs_norm,
                        f"pct<= {small_thr}": pct_small,
                        f"pct<= {medium_thr}": pct_med,
                    }
                )

    records.sort(key=lambda r: r["mean_abs_norm"])
    return records[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for near-equalities between engineered input/target features.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/CUP/ML-CUP25-TR.csv"),
        help="Path to CUP training CSV (default: data/CUP/ML-CUP25-TR.csv)",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Number of best relations to display (default: 15)")
    parser.add_argument(
        "--thr-small",
        type=float,
        default=0.1,
        help="Threshold for tight residual fraction on normalized residuals (default: 0.1)",
    )
    parser.add_argument(
        "--thr-med",
        type=float,
        default=0.5,
        help="Threshold for medium residual fraction on normalized residuals (default: 0.5)",
    )
    args = parser.parse_args()

    X, y = load_cup(args.file)
    input_feats = build_features(X, "input")
    target_feats = build_features(y, "target")

    best = compute_near_equalities(
        inputs=input_feats,
        targets=target_feats,
        thresholds=(args.thr_small, args.thr_med),
        top_n=args.top_n,
    )

    print(f"Top {len(best)} candidate near-equalities (sorted by mean abs residual):")
    for rec in best:
        print(
            f"{rec['relation']}: "
            f"mean_abs={rec['mean_abs']:.4f}, "
            f"median_abs={rec['median_abs']:.4f}, "
            f"max_abs={rec['max_abs']:.4f}, "
            f"pct<= {args.thr_small}={rec[f'pct<= {args.thr_small}']:.3f}, "
            f"pct<= {args.thr_med}={rec[f'pct<= {args.thr_med}']:.3f}"
        )


if __name__ == "__main__":
    main()
