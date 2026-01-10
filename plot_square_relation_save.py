import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont


def load_columns(file_path: Path):
    x6_vals = []
    y3_vals = []
    y4_vals = []
    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                # row: ID, x1..x12, y1, y2
                x6_vals.append(float(row[6]))
                y3_vals.append(float(row[13]))
                y4_vals.append(float(row[14]))
            except (ValueError, IndexError):
                continue
    return x6_vals, y3_vals, y4_vals


def _linear_map(value: float, vmin: float, vmax: float, pmin: float, pmax: float) -> float:
    if vmax == vmin:
        return (pmin + pmax) / 2
    return pmin + (value - vmin) * (pmax - pmin) / (vmax - vmin)


def _histogram(data, bins: int):
    if not data:
        return [0] * bins, [0.0] * (bins + 1)
    d_min, d_max = min(data), max(data)
    if d_min == d_max:
        d_min -= 1.0
        d_max += 1.0
    bin_edges = [d_min + i * (d_max - d_min) / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in data:
        idx = int((v - d_min) / (d_max - d_min) * bins)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, bin_edges


def make_plot(x6_vals, y3_vals, y4_vals, scale: float, out_path: Path) -> None:
    square_x6 = [v * v for v in x6_vals]
    diff_y = [a - b for a, b in zip(y3_vals, y4_vals)]
    square_diff_y = [v * v for v in diff_y]
    scaled_square_diff = [scale * v for v in square_diff_y]
    residual = [a - b for a, b in zip(square_x6, scaled_square_diff)]

    width, height = 1300, 520
    scatter_box = (70, 60, 650, 470)  # left, top, right, bottom
    hist_box = (720, 80, 1250, 470)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Scatter plot
    x_max = max(max(scaled_square_diff), max(square_x6), 1e-6) * 1.05
    x_min = 0.0
    y_max = x_max
    y_min = 0.0
    sx0, sy0, sx1, sy1 = scatter_box
    draw.rectangle(scatter_box, outline="lightgray")
    # y = x guideline
    for t_idx in range(200):
        t = x_min + (x_max - x_min) * t_idx / 199
        px = _linear_map(t, x_min, x_max, sx0, sx1)
        py = _linear_map(t, y_min, y_max, sy1, sy0)
        draw.point((px, py), fill="lightgray")
    # points
    for xr, sx in zip(scaled_square_diff, square_x6):
        px = _linear_map(xr, x_min, x_max, sx0, sx1)
        py = _linear_map(sx, y_min, y_max, sy1, sy0)
        draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=(44, 130, 201))
    draw.text((sx0, sy1 + 8), f"{scale:.4f} * square_y_3_minus_y_4", fill="black", font=font)
    draw.text((sx0 - 6, sy0 - 18), "square_x_6", fill="black", font=font)
    draw.text((sx0, sy0 - 34), "square_x_6 vs scaled square_y_3_minus_y_4", fill="black", font=font)

    # Histogram
    bins = 50
    hist, edges = _histogram(residual, bins)
    h_left, h_top, h_right, h_bottom = hist_box
    draw.rectangle(hist_box, outline="lightgray")
    max_count = max(hist) if hist else 1
    for idx, count in enumerate(hist):
        x0 = _linear_map(edges[idx], edges[0], edges[-1], h_left, h_right)
        x1 = _linear_map(edges[idx + 1], edges[0], edges[-1], h_left, h_right)
        bar_top = _linear_map(count, 0, max_count, h_bottom, h_top)
        draw.rectangle((x0, bar_top, x1, h_bottom), fill=(236, 112, 22))
    zero_x = _linear_map(0.0, edges[0], edges[-1], h_left, h_right)
    draw.line((zero_x, h_top, zero_x, h_bottom), fill="gray", width=1)
    mean_res = mean(residual) if residual else 0.0
    median_res = median(residual) if residual else 0.0
    summary = f"mean={mean_res:.4f}\nmedian={median_res:.4f}"
    draw.text((h_right - 130, h_top + 8), summary, fill="black", font=font)
    draw.text((h_left, h_top - 24), "Residual: square_x_6 - scaled square_y_3_minus_y_4", fill="black", font=font)
    draw.text((h_left, h_bottom + 8), "Residual", fill="black", font=font)
    draw.text((h_left - 6, h_top - 40), "Count", fill="black", font=font)

    title = "Relation: square_x_6 ~ scale * square_y_3_minus_y_4"
    draw.text((width // 2 - len(title) * 3, 20), title, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a plot for square_x_6 vs scaled square_y_3_minus_y_4."
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
        default=0.0013,
        help="Scale applied to square_y_3_minus_y_4 (default: 0.0013)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/square_x6_vs_scaled_square_y3y4.png"),
        help="Output image path (default: plots/square_x6_vs_scaled_square_y3y4.png)",
    )
    args = parser.parse_args()

    x6_vals, y3_vals, y4_vals = load_columns(args.file)
    make_plot(x6_vals, y3_vals, y4_vals, args.scale, args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
