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
    pairs = pair_names(arr.shape[1])
    for i, j in pairs:
        pfx = f"{prefix}_{i+1}_{j+1}"
        a_i, a_j = arr[:, i], arr[:, j]
        feats[f"prod_{pfx}"] = a_i * a_j
        feats[f"sum_{pfx}"] = a_i + a_j
        feats[f"diff_{pfx}"] = a_i - a_j
        feats[f"radius_{pfx}"] = np.hypot(a_i, a_j)
    # Angle only for first two dimensions (common case)
    if arr.shape[1] >= 2:
        angles = np.arctan2(arr[:, 1], arr[:, 0])
        feats[f"angle_{prefix}_1_2"] = angles
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
            else:
                residual = np.abs(vals_t - vals_x)
            mean_abs = float(residual.mean())
            median_abs = float(np.median(residual))
            max_abs = float(residual.max())
            pct_small = float((residual <= small_thr).mean())
            pct_med = float((residual <= medium_thr).mean())
            records.append(
                {
                    "relation": f"{name_t} ~ {name_x}",
                    "mean_abs": mean_abs,
                    "median_abs": median_abs,
                    "max_abs": max_abs,
                    f"pct<= {small_thr}": pct_small,
                    f"pct<= {medium_thr}": pct_med,
                }
            )

    records.sort(key=lambda r: r["mean_abs"])
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
        help="Threshold for tight residual fraction (default: 0.1)",
    )
    parser.add_argument(
        "--thr-med",
        type=float,
        default=0.5,
        help="Threshold for medium residual fraction (default: 0.5)",
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
