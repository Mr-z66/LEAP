import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


SIGNALS = {
    "low_entropy": ("final_entropy", "lowest", "Lowest entropy"),
    "low_mean_entropy": ("mean_entropy", "lowest", "Lowest mean entropy"),
    "low_max_entropy": ("max_entropy", "lowest", "Lowest max entropy"),
    "high_top1": ("final_top1_prob", "highest", "Highest top-1 prob."),
    "high_margin": ("final_margin", "highest", "Highest margin"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot correct/wrong donut charts inside high-confidence surface-signal slices."
    )
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument(
        "--signal",
        choices=sorted(SIGNALS),
        default="low_entropy",
        help="Surface confidence slice to analyze.",
    )
    parser.add_argument(
        "--slices",
        default="0.10,0.20,0.30",
        help="Comma-separated fractions of chunks to keep, e.g. 0.10,0.20,0.30.",
    )
    parser.add_argument("--output-dir", default="result/analysis_outputs/preliminary_figures")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def parse_fractions(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one slice fraction.")
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Slice fractions must be in (0, 1]: {value}")
    return values


def scalar_value(row, key):
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing signal field: {key}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def load_arrays(path, label_key, signal):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    field, _, _ = SIGNALS[signal]
    scores = []
    labels = []
    for row in rows:
        label = int(row.get(label_key, -1))
        if label not in {0, 1}:
            continue
        scores.append(scalar_value(row, field))
        labels.append(label)
    if not scores:
        raise ValueError(f"No valid rows found for label_key={label_key!r}.")
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def summarize_slices(scores, labels, signal, fractions):
    _, direction, title = SIGNALS[signal]
    order = np.argsort(scores)
    if direction == "highest":
        order = order[::-1]

    summaries = []
    for fraction in fractions:
        count = max(1, int(np.ceil(fraction * len(order))))
        selected = order[:count]
        selected_labels = labels[selected]
        correct_count = int(np.sum(selected_labels == 1))
        wrong_count = int(np.sum(selected_labels == 0))
        summaries.append(
            {
                "signal": signal,
                "slice_title": f"{title} {int(round(fraction * 100))}%",
                "slice_fraction": fraction,
                "total_count": int(count),
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "correct_rate": correct_count / count,
                "wrong_rate": wrong_count / count,
            }
        )
    return summaries


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_donuts(summaries, output_png, output_pdf):
    fig, axes = plt.subplots(1, len(summaries), figsize=(2.65 * len(summaries), 2.75))
    if len(summaries) == 1:
        axes = [axes]

    colors = ["#2F6DA3", "#D94136"]
    for ax, summary in zip(axes, summaries):
        values = [summary["correct_count"], summary["wrong_count"]]
        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
        )
        ax.text(
            0,
            0.08,
            f"{summary['wrong_rate'] * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#D94136",
        )
        ax.text(0, -0.16, "wrong", ha="center", va="center", fontsize=8, color="#D94136")
        ax.set_title(summary["slice_title"], fontsize=10, pad=7)
        ax.text(
            0,
            -1.18,
            f"n={summary['total_count']}",
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )

    fig.legend(
        wedges,
        ["correct", "wrong"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.08, wspace=0.2)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fractions = parse_fractions(args.slices)
    scores, labels = load_arrays(args.label_path, args.label_key, args.signal)
    summaries = summarize_slices(scores, labels, args.signal, fractions)

    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]
    csv_path = os.path.join(args.output_dir, f"{output_prefix}_{args.signal}_confident_slices.csv")
    png_path = os.path.join(args.output_dir, f"{output_prefix}_{args.signal}_confident_slices.png")
    pdf_path = os.path.join(args.output_dir, f"{output_prefix}_{args.signal}_confident_slices.pdf")

    write_csv(csv_path, summaries)
    plot_donuts(summaries, png_path, pdf_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    for summary in summaries:
        print(
            f"{summary['slice_title']}: wrong={summary['wrong_rate'] * 100:.2f}% "
            f"({summary['wrong_count']}/{summary['total_count']})"
        )


if __name__ == "__main__":
    main()
