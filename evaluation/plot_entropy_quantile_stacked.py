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

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    correct_color = "#3F78A8"
    wrong_color = "#D84A3A"
    edge_color = "#2F2F2F"

    ax.bar(x, correct, color=correct_color, edgecolor=edge_color, linewidth=0.45, label="correct")
    ax.bar(x, wrong, bottom=correct, color=wrong_color, edgecolor=edge_color, linewidth=0.45, label="wrong")
    ax.axhline(100.0 - overall_wrong, color="#444444", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.text(
        len(summaries) - 0.45,
        100.0 - overall_wrong + 1.2,
        f"overall wrong rate = {overall_wrong:.1f}%",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    for idx, (correct_rate, wrong_rate, count) in enumerate(zip(correct, wrong, counts)):
        if wrong_rate >= 4.0:
            ax.text(idx, correct_rate + wrong_rate / 2.0, f"{wrong_rate:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(idx, 2.0, f"n={count}", ha="center", va="bottom", fontsize=7, color="white", rotation=90, alpha=0.85)

    ax.set_title(f"Correct/wrong chunks across {signal_title.lower()} quantiles", fontsize=12, pad=9)
    ax.set_ylabel("Chunk composition (%)", fontsize=10)
    ax.set_xlabel("Entropy quantile bins (low to high)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([row["bin"] for row in summaries], fontsize=9)
    ax.tick_params(axis="y", labelsize=9, length=2)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.32)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.2)
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
