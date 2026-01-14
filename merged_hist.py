from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import merged


def main() -> None:
    runs = 20
    metrics = {
        "mee_de": [],
        "mee_z_est": [],
        "mee_z_ref": [],
        "mee_ens_refined": [],
        "mee_selected": [],
    }

    for _ in range(runs):
        result = merged.main(do_plots=False)
        for key in metrics:
            metrics[key].append(result[key])

    print("Average MEE over runs:")
    for key, values in metrics.items():
        avg = float(np.mean(values))
        print(f"{key}: {avg:.5f}")

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, values in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins=20, color="tab:blue", alpha=0.8)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        ax.set_title(f"{key} over {runs} runs")
        fig.tight_layout()
        out_path = out_dir / f"{key}_hist.png"
        fig.savefig(out_path, dpi=150)
        print(out_path)


if __name__ == "__main__":
    main()
