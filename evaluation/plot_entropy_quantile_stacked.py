import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


SIGNALS = {
    "entropy": ("final_entropy", "Final-token entropy"),
    "mean_entropy": ("mean_entropy", "Chunk-averaged entropy"),
    "max_entropy": ("max_entropy", "Maximum chunk entropy"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot correct/wrong composition across entropy quantile bins."
    )
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument("--signal", choices=sorted(SIGNALS), default="mean_entropy")
    parser.add_argument("--bins", type=int, default=10, help="Number of entropy quantile bins.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap after loading.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for --max-rows.")
    parser.add_argument("--output-dir", default="result/analysis_outputs/preliminary_figures")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def scalar_value(row, key):
    import torch

    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing signal field: {key}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def load_arrays(path, label_key, signal, max_rows, seed):
    import torch

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if max_rows is not None and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(np.arange(len(rows)), size=max_rows, replace=False))
        rows = [rows[int(index)] for index in indices]

    field, _ = SIGNALS[signal]
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


def quantile_summaries(scores, labels, bins):
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
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
        selected = labels[mask]
        correct = int(np.sum(selected == 1))
        wrong = int(np.sum(selected == 0))
        count = correct + wrong
        summaries.append(
            {
                "bin": f"Q{len(summaries) + 1}",
                "left": float(left),
                "right": float(right),
                "count": count,
                "correct_count": correct,
                "wrong_count": wrong,
                "correct_rate": correct / count,
                "wrong_rate": wrong / count,
            }
        )
    return summaries


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_stacked(summaries, signal_title, output_png, output_pdf):
    x = np.arange(len(summaries))
    correct = np.asarray([row["correct_rate"] * 100.0 for row in summaries])
    wrong = np.asarray([row["wrong_rate"] * 100.0 for row in summaries])
    counts = [row["count"] for row in summaries]
    overall_wrong = sum(row["wrong_count"] for row in summaries) / sum(row["count"] for row in summaries) * 100.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    correct_color = "#4C78A8"
    wrong_color = "#E45756"
    edge_color = "white"

    width = 0.74
    ax.bar(x, correct, width=width, color=correct_color, edgecolor=edge_color, linewidth=0.5, label="Correct")
    ax.bar(x, wrong, width=width, bottom=correct, color=wrong_color, edgecolor=edge_color, linewidth=0.5, label="Wrong")
    ax.axhline(100.0 - overall_wrong, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(
        len(summaries) - 0.08,
        100.0 - overall_wrong + 0.9,
        f"overall wrong={overall_wrong:.1f}%",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )

    for idx, (correct_rate, wrong_rate, count) in enumerate(zip(correct, wrong, counts)):
        if wrong_rate >= 4.0:
            ax.text(
                idx,
                correct_rate + wrong_rate / 2.0,
                f"{wrong_rate:.1f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white",
                fontweight="bold",
            )
        ax.text(
            idx,
            3.0,
            str(count),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="white",
            alpha=0.82,
        )

    ax.set_title(f"Chunk correctness across {signal_title.lower()} quantiles", pad=8)
    ax.set_ylabel("Composition (%)")
    ax.set_xlabel("Entropy quantiles (low to high)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.65, len(summaries) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([row["bin"] for row in summaries])
    ax.tick_params(axis="both", length=2.2, width=0.7)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handlelength=1.3,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)

    fig.text(0.985, 0.015, "Numbers inside bars denote wrong rate; bottom numbers denote bin size.", ha="right", va="bottom", fontsize=5.8, color="#555555")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.24)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scores, labels = load_arrays(args.label_path, args.label_key, args.signal, args.max_rows, args.seed)
    summaries = quantile_summaries(scores, labels, args.bins)

    prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]
    csv_path = os.path.join(args.output_dir, f"{prefix}_{args.signal}_quantile_composition.csv")
    png_path = os.path.join(args.output_dir, f"{prefix}_{args.signal}_quantile_composition.png")
    pdf_path = os.path.join(args.output_dir, f"{prefix}_{args.signal}_quantile_composition.pdf")

    write_csv(csv_path, summaries)
    plot_stacked(summaries, SIGNALS[args.signal][1], png_path, pdf_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print("Wrong rates by entropy quantile:")
    for row in summaries:
        print(f"{row['bin']}: {row['wrong_rate'] * 100:.2f}% ({row['wrong_count']}/{row['count']})")


if __name__ == "__main__":
    main()
