import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


FILE_PATH = "data/CUP/ML-CUP25-TR.csv"
W_THEORETICAL = (1 + 1 / np.sqrt(2)) / 3
GRID_SIZE = 5000
K_NEIGHBORS = 4999
TOP_PAIRS = 2000
RATIO_THRESH = 0.25


def get_physical_features(z_arr):
    if np.isscalar(z_arr):
        z_arr = np.array([z_arr])
    feats = [z_arr]
    for k in range(1, 10):
        feats.append(np.sin(k * W_THEORETICAL * z_arr))
        feats.append(np.cos(k * W_THEORETICAL * z_arr))
    return np.column_stack(feats)


def main():
    df = pd.read_csv(FILE_PATH, comment="#", header=None)
    X = df.iloc[:, 1:13].values
    Y = df.iloc[:, 13:17].values
    z_all = Y[:, 2] - Y[:, 3]

    z_min, z_max = float(z_all.min()), float(z_all.max())
    z_grid = np.linspace(z_min, z_max, GRID_SIZE)

    train_feats = get_physical_features(z_all)
    models = []
    for i in range(12):
        reg = LinearRegression().fit(train_feats, X[:, i])
        models.append(reg)

    coefs_mat = np.array([m.coef_ for m in models])
    intercepts = np.array([m.intercept_ for m in models])

    grid_feats = get_physical_features(z_grid)
    X_grid = grid_feats @ coefs_mat.T + intercepts

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm="ball_tree").fit(X_grid)
    dists, indices = nbrs.kneighbors(X_grid)

    pairs = []
    for i in range(len(z_grid)):
        for dist_x, j in zip(dists[i, 1:], indices[i, 1:]):
            if j <= i:
                continue
            dist_z = abs(z_grid[i] - z_grid[j])
            # print(dist_z)
            if dist_z < 1.0:
                continue
            ratio = dist_x / dist_z
            pairs.append((ratio, i, j, dist_x, dist_z))

    pairs.sort(key=lambda x: x[0])
    top_pairs = pairs[:TOP_PAIRS]
    filtered_pairs = [p for p in pairs if p[0] <= RATIO_THRESH]

    out_dir = Path("plots/bench5")
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_df(rows):
        return pd.DataFrame(
            [
                {
                    "ratio": r,
                    "z_i": float(z_grid[i]),
                    "z_j": float(z_grid[j]),
                    "dist_x": float(dx),
                    "dist_z": float(dz),
                }
                for r, i, j, dx, dz in rows
            ]
        )

    out_path = out_dir / "closest_pairs_by_ratio.csv"
    to_df(top_pairs).to_csv(out_path, index=False)
    print(f"Saved top pairs: {out_path}")

    out_path = out_dir / "pairs_below_ratio_thresh.csv"
    to_df(filtered_pairs).to_csv(out_path, index=False)
    print(f"Saved threshold pairs: {out_path}")

    print(f"Total pairs evaluated: {len(pairs)}")
    print(f"Pairs with ratio <= {RATIO_THRESH}: {len(filtered_pairs)}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time() - start:.2f}s")
