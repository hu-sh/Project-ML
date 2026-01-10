import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


FeatureDict = Dict[str, np.ndarray]


def load_cup(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path, comment="#", header=None)
    values = df.iloc[:, 1:]
    X = values.iloc[:, :12].to_numpy()
    Y = values.iloc[:, 12:].to_numpy()
    return X, Y


def pair_names(n_cols: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]


def build_base_matrix(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    cols = []
    names: List[str] = []
    for i in range(X.shape[1]):
        cols.append(X[:, i])
        names.append(f"x_{i+1}")
    if Y.shape[1] < 4:
        raise ValueError("Expected at least 4 target columns to build y_3_minus_y_4 feature.")
    mean_abs_x = np.mean(np.abs(X), axis=1)
    diff_y = Y[:, 2] - Y[:, 3]
    diff_abs_mean = float(np.mean(np.abs(diff_y))) if diff_y.size else 0.0
    mean_abs_x_mean = float(mean_abs_x.mean()) if mean_abs_x.size else 0.0
    scale = mean_abs_x_mean / max(diff_abs_mean, 1e-12) if diff_abs_mean != 0 else 1.0
    scaled_diff = scale * diff_y
    cols.append(scaled_diff)
    names.append("scaled_y_3_minus_y_4")
    cols.append(Y[:, 0])
    names.append("y_1")
    cols.append(Y[:, 1])
    names.append("y_2")
    cols.append(mean_abs_x)
    names.append("mean_abs_x")
    return np.column_stack(cols), names


def build_features(arr: np.ndarray, base_names: List[str], trig_freqs: Sequence[float]) -> FeatureDict:
    feats: FeatureDict = {}
    eps = 1e-12
    # Single-variable features
    for idx, name in enumerate(base_names):
        col = arr[:, idx]
        feats[f"raw_{name}"] = col
        feats[f"abs_{name}"] = np.abs(col)
        feats[f"sign_{name}"] = np.sign(col)
        feats[f"square_{name}"] = col**2
        feats[f"sqrtabs_{name}"] = np.abs(col) ** 0.5
        feats[f"logabs_{name}"] = np.log(np.abs(col) + eps)

    # Aggregates across all variables
    geom_all = np.exp(np.mean(np.log(np.abs(arr) + eps), axis=1))
    mean_abs_all = np.mean(np.abs(arr), axis=1)
    l2_all = np.linalg.norm(arr, axis=1)
    max_abs_all = np.max(np.abs(arr), axis=1)
    std_abs_all = np.std(np.abs(arr), axis=1)
    feats["geom_all"] = geom_all
    feats["meanabs_all"] = mean_abs_all
    feats["l2_all"] = l2_all
    feats["maxabs_all"] = max_abs_all
    feats["stdabs_all"] = std_abs_all

    # Pairwise features
    for i, j in pair_names(arr.shape[1]):
        a_i, a_j = arr[:, i], arr[:, j]
        pfx = f"{base_names[i]}_{base_names[j]}"
        feats[f"prod_{pfx}"] = a_i * a_j
        feats[f"absprod_{pfx}"] = np.abs(a_i * a_j)
        feats[f"sum_{pfx}"] = a_i + a_j
        feats[f"diff_{pfx}"] = a_i - a_j
        feats[f"abssum_{pfx}"] = np.abs(a_i + a_j)
        feats[f"absdiff_{pfx}"] = np.abs(a_i - a_j)
        feats[f"radius_{pfx}"] = np.hypot(a_i, a_j)
        feats[f"angle_{pfx}"] = np.arctan2(a_j, a_i)
        feats[f"geom_{pfx}"] = np.sqrt(np.abs(a_i * a_j))
        feats[f"ratio_{pfx}"] = a_i / (a_j + eps)
        feats[f"ratio_inv_{pfx}"] = a_j / (a_i + eps)
        feats[f"normdiff_{pfx}"] = (a_i - a_j) / (a_i + a_j)

    # Trigonometric combinations of scaled (y3 - y4)
    if "scaled_y_3_minus_y_4" in base_names:
        idx = base_names.index("scaled_y_3_minus_y_4")
        diff_vals = arr[:, idx]
        for freq in trig_freqs:
            feats[f"trigcos_{freq:.4f}_scaled_y_3_minus_y_4"] = diff_vals * np.cos(freq * diff_vals)
            feats[f"trigsin_{freq:.4f}_scaled_y_3_minus_y_4"] = diff_vals * np.sin(freq * diff_vals)

    return feats


def angle_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, 2 * math.pi - raw)


def compute_pairwise_relations(
    features: FeatureDict, thresholds: Tuple[float, float], top_n: int
) -> List[Dict[str, float]]:
    small_thr, medium_thr = thresholds
    items = list(features.items())
    records: List[Dict[str, float]] = []

    def _match_base(base_a: str, base_b: str) -> bool:
        if base_a == base_b:
            return True
        trig_bases = {"trigcos", "trigsin"}
        if (base_a == "raw" and base_b in trig_bases) or (base_b == "raw" and base_a in trig_bases):
            return True
        if (base_a == "radius" and base_b == "absdiff") or (base_a == "absdiff" and base_b == "radius"):
            return True
        return False

    for idx_a, (name_a, vals_a) in enumerate(items):
        base_a = name_a.split("_", 1)[0]
        for name_b, vals_b in items[idx_a + 1 :]:
            base_b = name_b.split("_", 1)[0]
            if not _match_base(base_a, base_b):
                continue

            if base_a == "angle":
                residual = angle_residual(vals_a, vals_b)
                candidates = [("1.0", residual)]
            else:
                residual_unscaled = np.abs(vals_a - vals_b)
                denom = float(np.dot(vals_b, vals_b))
                if denom > 0:
                    scale = float(np.dot(vals_a, vals_b) / denom)
                    residual_scaled = np.abs(vals_a - scale * vals_b)
                    candidates = [("1.0", residual_unscaled), (f"{scale:.4f}", residual_scaled)]
                else:
                    candidates = [("1.0", residual_unscaled)]

            for scale_str, residual in candidates:
                denom_norm = np.maximum(np.maximum(np.abs(vals_a), np.abs(vals_b)), 1e-6)
                residual_norm = residual / denom_norm

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
                        "relation": f"{name_a} ~ {scale_str} * {name_b}",
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
    parser = argparse.ArgumentParser(
        description="Scan for near-equalities among engineered selected features (raw inputs, y3-y4, mean abs input)."
    )
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

    X, Y = load_cup(args.file)
    base_matrix, base_names = build_base_matrix(X, Y)
    trig_freqs = (1.1395, 0.5, 1.0, 1.5, 2.0)
    engineered_features = build_features(base_matrix, base_names, trig_freqs)

    best = compute_pairwise_relations(
        features=engineered_features,
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
