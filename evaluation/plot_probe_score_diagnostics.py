import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diagnostic figures showing hidden-state probe scores separate correct and wrong chunks."
    )
    parser.add_argument("--score-csv", required=True, help="CSV from evaluation.export_probe_chunk_scores.")
    parser.add_argument("--score-column", default="trigger_score", help="Score column where higher means more likely wrong.")
    parser.add_argument("--label-column", default="label", help="Label column. label=1 means correct; label=0 means wrong.")
    parser.add_argument("--bins", type=int, default=10, help="Number of score quantile bins.")
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


def load_scores(path, score_column, label_column):
    scores = []
    labels = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(float(row[label_column]))
            if label not in {0, 1}:
                continue
            scores.append(float(row[score_column]))
            labels.append(label)
    if not scores:
        raise ValueError(f"No valid rows in {path}")
    labels = np.asarray(labels, dtype=np.int64)
    return np.asarray(scores, dtype=np.float64), labels, 1 - labels


def quantile_summaries(scores, error_labels, bins):
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(float(np.min(scores)), float(np.max(scores)), bins + 1)

    rows = []
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
        rows.append(
            {
                "bin": f"Q{len(rows) + 1}",
                "left": float(left),
                "right": float(right),
                "count": count,
                "wrong_count": wrong,
                "wrong_rate": rate,
                "wrong_rate_se": float(stderr),
            }
        )
    return rows


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


def plot_ecdf(scores, labels, output_base):
    correct = np.sort(scores[labels == 1])
    wrong = np.sort(scores[labels == 0])
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for values, color, name in [(correct, "#4C78A8", "Correct"), (wrong, "#E45756", "Wrong")]:
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=color, linewidth=1.8, label=name)
    ax.set_title("Probe scores shift wrong chunks to higher risk")
    ax.set_xlabel("Hidden-state probe error score")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.18)
    save_figure(fig, output_base)


def plot_violin(scores, labels, output_base):
    correct = scores[labels == 1]
    wrong = scores[labels == 0]
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    parts = ax.violinplot([correct, wrong], positions=[0, 1], widths=0.72, showmeans=False, showmedians=True)
    for body, color in zip(parts["bodies"], ["#4C78A8", "#E45756"]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.45)
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
    ax.set_ylabel("Hidden-state probe error score")
    ax.set_title("Wrong chunks receive higher probe risk")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.17)
    save_figure(fig, output_base)


def plot_wrong_rate_curve(summaries, output_base):
    x = np.arange(len(summaries))
    y = np.asarray([row["wrong_rate"] * 100.0 for row in summaries])
    se = np.asarray([row["wrong_rate_se"] * 100.0 for row in summaries])
    overall = sum(row["wrong_count"] for row in summaries) / sum(row["count"] for row in summaries) * 100.0

    fig, ax = plt.subplots(figsize=(4.4, 2.45))
    ax.plot(x, y, marker="o", markersize=4.2, linewidth=1.7, color="#E45756")
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color="#E45756", alpha=0.14, linewidth=0)
    ax.axhline(overall, color="#555555", linestyle="--", linewidth=0.8, label=f"Overall={overall:.1f}%")
    ax.set_title("Error rate rises with hidden-state probe risk")
    ax.set_xlabel("Probe score quantiles (low to high risk)")
    ax.set_ylabel("Wrong chunk rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([row["bin"] for row in summaries])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.85, bottom=0.2)
    save_figure(fig, output_base)


def plot_roc_pr(scores, error_labels, output_base):
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
    fig.suptitle("Hidden-state probe as an error detector", y=1.02, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.22, wspace=0.35)
    save_figure(fig, output_base)


def plot_two_panel(scores, labels, summaries, output_base):
    correct = scores[labels == 1]
    wrong = scores[labels == 0]
    x = np.arange(len(summaries))
    y = np.asarray([row["wrong_rate"] * 100.0 for row in summaries])
    overall = sum(row["wrong_count"] for row in summaries) / sum(row["count"] for row in summaries) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5))
    axes[0].hist(correct, bins=24, density=True, alpha=0.55, color="#4C78A8", label="Correct")
    axes[0].hist(wrong, bins=24, density=True, alpha=0.55, color="#E45756", label="Wrong")
    axes[0].set_title("Score distributions")
    axes[0].set_xlabel("Probe error score")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, loc="upper center")

    axes[1].plot(x, y, marker="o", markersize=4.0, linewidth=1.6, color="#E45756")
    axes[1].axhline(overall, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_title("Error rate by probe score")
    axes[1].set_xlabel("Probe score quantiles")
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
    scores, labels, error_labels = load_scores(args.score_csv, args.score_column, args.label_column)
    summaries = quantile_summaries(scores, error_labels, args.bins)

    prefix = args.output_prefix or os.path.splitext(os.path.basename(args.score_csv))[0]
    base = os.path.join(args.output_dir, f"{prefix}_probe_score")
    csv_path = f"{base}_diagnostics.csv"
    write_csv(csv_path, summaries)
    print(f"Wrote: {csv_path}")

    plot_ecdf(scores, labels, f"{base}_ecdf")
    plot_violin(scores, labels, f"{base}_violin")
    plot_wrong_rate_curve(summaries, f"{base}_wrong_rate_curve")
    plot_roc_pr(scores, error_labels, f"{base}_roc_pr")
    plot_two_panel(scores, labels, summaries, f"{base}_two_panel")


if __name__ == "__main__":
    main()
