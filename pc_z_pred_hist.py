from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INTERCEPT = -7.4426
BETAS = np.array([-10.9003, 2.6545])


def main() -> None:
    df = pd.read_csv("data/CUP/ML-CUP25-TR-pca2.csv")
    if not {"pc1", "pc2", "t3", "t4"}.issubset(df.columns):
        raise SystemExit("Expected columns: pc1, pc2, t3, t4")

    pc = df[["pc1", "pc2"]].to_numpy()
    z_true = (df["t3"] - df["t4"]).to_numpy()
    z_pred = INTERCEPT + pc @ BETAS

    err = z_pred - z_true

    out_path = Path("plots/z_pred_pc_hist.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(err, bins=50, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Error (z_pred - z_true)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of z_pred error (linear PC model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    out_path = Path("plots/z_pred_vs_z_true.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(z_true, z_pred, s=16, alpha=0.7, color="tab:purple")
    lim = max(np.max(np.abs(z_true)), np.max(np.abs(z_pred)))
    ax.plot([-lim, lim], [-lim, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("z_true")
    ax.set_ylabel("z_pred")
    ax.set_title("z_pred vs z_true (linear PC model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
