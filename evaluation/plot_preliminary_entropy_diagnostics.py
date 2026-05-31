import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


SIGNALS = {
    "entropy": ("final_entropy", "Final-token entropy"),
    "mean_entropy": ("mean_entropy", "Chunk-averaged entropy"),
    "max_entropy": ("max_entropy", "Maximum chunk entropy"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot preliminary diagnostics showing entropy is a weak chunk-correctness signal."
    )
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument("--signal", choices=sorted(SIGNALS), default="mean_entropy")
    parser.add_argument("--bins", type=int, default=10, help="Number of entropy quantile bins.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional random row cap after loading.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for --max-rows.")
    parser.add_argument("--output-dir", default="result/analysis_outputs/preliminary_figures")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def apply_paper_style():
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
    labels = np.asarray(labels, dtype=np.int64)
    return np.asarray(scores, dtype=np.float64), labels, 1 - labels


def quantile_summaries(scores, error_labels, bins):
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
        selected = error_labels[mask]
        count = int(mask.sum())
        wrong = int(selected.sum())
        rate = wrong / count
        stderr = np.sqrt(max(rate * (1.0 - rate), 0.0) / count)
        summaries.append(
            {
                "bin": f"Q{len(summaries) + 1}",
                "left": float(left),
                "right": float(right),
                "center": float(np.mean(scores[mask])),
                "count": count,
                "wrong_count": wrong,
                "wrong_rate": rate,
                "wrong_rate_se": float(stderr),
            }
        )
    return summaries


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output_base):
    png_path = f"{output_base}.png"
    pdf_path = f"{output_base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


def plot_ecdf(scores, labels, signal_title, output_base):
    correct = np.sort(scores[labels == 1])
    wrong = np.sort(scores[labels == 0])
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for values, color, name in [(correct, "#4C78A8", "Correct"), (wrong, "#E45756", "Wrong")]:
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=color, linewidth=1.8, label=name)
    ax.set_title("Entropy distributions nearly overlap")
    ax.set_xlabel(signal_title)
    ax.set_ylabel("Cumulative fraction")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.18)
    save_figure(fig, output_base)


def plot_violin(scores, labels, signal_title, output_base):
    correct = scores[labels == 1]
    wrong = scores[labels == 0]
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    parts = ax.violinplot([correct, wrong], positions=[0, 1], widths=0.72, showmeans=False, showmedians=True)
    for body, color in zip(parts["bodies"], ["#4C78A8", "#E45756"]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.42)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#333333")
        parts[key].set_linewidth(0.8)
    ax.boxplot(
        [correct, wrong],
        positions=[0, 1],
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.7},
        medianprops={"color": "#333333", "linewidth": 0.9},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Wrong"])
    ax.set_ylabel(signal_title)
    ax.set_title("Correct and wrong chunks share similar entropy")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.17)
    save_figure(fig, output_base)


def plot_wrong_rate_curve(summaries, signal_title, output_base):
    x = np.arange(len(summaries))
    y = np.asarray([row["wrong_rate"] * 100.0 for row in summaries])
    se = np.asarray([row["wrong_rate_se"] * 100.0 for row in summaries])
    overall = sum(row["wrong_count"] for row in summaries) / sum(row["count"] for row in summaries) * 100.0

    fig, ax = plt.subplots(figsize=(4.4, 2.45))
    ax.plot(x, y, marker="o", markersize=4.2, linewidth=1.7, color="#E45756")
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color="#E45756", alpha=0.14, linewidth=0)
    ax.axhline(overall, color="#555555", linestyle="--", linewidth=0.8, label=f"Overall={overall:.1f}%")
    ax.set_title("Error rate is not monotonic with entropy")
    ax.set_xlabel("Entropy quantiles (low to high)")
    ax.set_ylabel("Wrong chunk rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([row["bin"] for row in summaries])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.85, bottom=0.2)
    save_figure(fig, output_base)


def plot_roc_pr(scores, error_labels, signal_title, output_base):
    fpr, tpr, _ = roc_curve(error_labels, scores)
    precision, recall, _ = precision_recall_curve(error_labels, scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(error_labels, scores)
    prevalence = float(np.mean(error_labels))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    axes[0].plot(fpr, tpr, color="#E45756", linewidth=1.8, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(recall, precision, color="#E45756", linewidth=1.8, label=f"AP={pr_auc:.3f}")
    axes[1].axhline(prevalence, color="#777777", linestyle="--", linewidth=0.8, label=f"Base={prevalence:.3f}")
    axes[1].set_title("Precision-recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{signal_title} as an error detector", y=1.02, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.22, wspace=0.35)
    save_figure(fig, output_base)


def plot_two_panel(scores, labels, summaries, signal_title, output_base):
    correct = np.sort(scores[labels == 1])
    wrong = np.sort(scores[labels == 0])
    x = np.arange(len(summaries))
    y = np.asarray([row["wrong_rate"] * 100.0 for row in summaries])
    overall = sum(row["wrong_count"] for row in summaries) / sum(row["count"] for row in summaries) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5))
    for values, color, name in [(correct, "#4C78A8", "Correct"), (wrong, "#E45756", "Wrong")]:
        ecdf_y = np.arange(1, len(values) + 1) / len(values)
        axes[0].step(values, ecdf_y, where="post", color=color, linewidth=1.7, label=name)
    axes[0].set_title("Distribution overlap")
    axes[0].set_xlabel(signal_title)
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(x, y, marker="o", markersize=4.0, linewidth=1.6, color="#E45756")
    axes[1].axhline(overall, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_title("Non-monotonic error rate")
    axes[1].set_xlabel("Entropy quantiles")
    axes[1].set_ylabel("Wrong rate (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([row["bin"] for row in summaries])

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.22, wspace=0.34)
    save_figure(fig, output_base)


def main():
    args = parse_args()
    apply_paper_style()
    os.makedirs(args.output_dir, exist_ok=True)
    scores, labels, error_labels = load_arrays(args.label_path, args.label_key, args.signal, args.max_rows, args.seed)
    summaries = quantile_summaries(scores, error_labels, args.bins)

    prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]
    base = os.path.join(args.output_dir, f"{prefix}_{args.signal}")
    csv_path = f"{base}_entropy_diagnostics.csv"
    write_csv(csv_path, summaries)
    print(f"Wrote: {csv_path}")

    signal_title = SIGNALS[args.signal][1]
    plot_ecdf(scores, labels, signal_title, f"{base}_ecdf")
    plot_violin(scores, labels, signal_title, f"{base}_violin")
    plot_wrong_rate_curve(summaries, signal_title, f"{base}_wrong_rate_curve")
    plot_roc_pr(scores, error_labels, signal_title, f"{base}_roc_pr")
    plot_two_panel(scores, labels, summaries, signal_title, f"{base}_two_panel")


if __name__ == "__main__":
    main()
