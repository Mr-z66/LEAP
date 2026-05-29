import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


SIGNALS = {
    "entropy": ("final_entropy", 1.0, "Entropy"),
    "mean_entropy": ("mean_entropy", 1.0, "Mean entropy"),
    "max_entropy": ("max_entropy", 1.0, "Max entropy"),
    "top1": ("final_top1_prob", 1.0, "Top-1 probability"),
    "margin": ("final_margin", 1.0, "Logit margin"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether surface uncertainty bins reliably indicate chunk errors."
    )
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument("--signals", default="entropy,top1,margin", help="Comma-separated signal names.")
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick plotting.")
    parser.add_argument("--output-dir", default="result/analysis_outputs/preliminary_figures")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def parse_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def scalar_value(row, key):
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing signal field: {key}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def load_rows(path, max_rows):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def collect_arrays(rows, label_key, signal_name):
    key, direction, _ = SIGNALS[signal_name]
    scores = []
    errors = []
    for row in rows:
        label = int(row.get(label_key, -1))
        if label not in {0, 1}:
            continue
        scores.append(direction * scalar_value(row, key))
        errors.append(1 - label)
    if not scores:
        raise ValueError(f"No valid rows found for label_key={label_key!r}")
    return np.asarray(scores, dtype=np.float64), np.asarray(errors, dtype=np.int64)


def quantile_bin_summary(scores, errors, bins):
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(scores, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(float(np.min(scores)), float(np.max(scores)), bins + 1)

    summaries = []
    for index in range(len(edges) - 1):
        left = edges[index]
        right = edges[index + 1]
        if index == len(edges) - 2:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not np.any(mask):
            continue
        bin_errors = errors[mask]
        summaries.append(
            {
                "bin_index": index,
                "left": float(left),
                "right": float(right),
                "center": float(np.mean(scores[mask])),
                "count": int(mask.sum()),
                "error_count": int(bin_errors.sum()),
                "error_rate": float(bin_errors.mean()),
            }
        )
    return summaries


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_signal_bins(all_summaries, output_png, output_pdf):
    fig, axes = plt.subplots(1, len(all_summaries), figsize=(3.0 * len(all_summaries), 2.65), sharey=True)
    if len(all_summaries) == 1:
        axes = [axes]

    for ax, item in zip(axes, all_summaries):
        summaries = item["summaries"]
        x = np.arange(len(summaries))
        y = [row["error_rate"] * 100.0 for row in summaries]
        labels = [f"Q{row['bin_index'] + 1}" for row in summaries]
        counts = [row["count"] for row in summaries]
        colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(summaries)))

        ax.bar(x, y, color=colors, edgecolor="#3A3A3A", linewidth=0.55)
        ax.set_title(item["title"], fontsize=10, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.tick_params(axis="y", labelsize=8, length=2)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.set_xlabel("Low to high signal", fontsize=8)
        for xi, yi, count in zip(x, y, counts):
            ax.text(xi, yi + 0.35, f"{yi:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(xi, 0.7, f"n={count}", ha="center", va="bottom", fontsize=6, rotation=90, alpha=0.65)

    axes[0].set_ylabel("Wrong chunk rate (%)", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.84, bottom=0.24, wspace=0.12)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    signal_names = parse_csv(args.signals)
    rows = load_rows(args.label_path, args.max_rows)

    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]
    all_csv_rows = []
    all_summaries = []
    for signal_name in signal_names:
        if signal_name not in SIGNALS:
            raise ValueError(f"Unknown signal {signal_name!r}. Available: {', '.join(sorted(SIGNALS))}")
        scores, errors = collect_arrays(rows, args.label_key, signal_name)
        summaries = quantile_bin_summary(scores, errors, args.bins)
        _, _, title = SIGNALS[signal_name]
        all_summaries.append({"signal": signal_name, "title": title, "summaries": summaries})
        for row in summaries:
            all_csv_rows.append({"signal": signal_name, **row})

    csv_path = os.path.join(args.output_dir, f"{output_prefix}_surface_signal_bins.csv")
    png_path = os.path.join(args.output_dir, f"{output_prefix}_surface_signal_bins.png")
    pdf_path = os.path.join(args.output_dir, f"{output_prefix}_surface_signal_bins.pdf")
    write_csv(csv_path, all_csv_rows)
    plot_signal_bins(all_summaries, png_path, pdf_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
