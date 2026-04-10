import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "analysis_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot scheduler trade-off curves from one or more scheduler summary JSON files.")
    parser.add_argument(
        "--summary-paths",
        nargs="+",
        required=True,
        help="One or more scheduler_run_summary.json paths.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels for each summary path. Defaults to file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Scheduler Tradeoff",
        help="Prefix used in chart titles.",
    )
    return parser.parse_args()


def load_summary(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("threshold_summaries", [])
    if not rows:
        raise ValueError(f"No threshold_summaries found in {path}")
    return rows


def infer_label(path):
    return Path(path).stem


def sort_rows(rows):
    return sorted(rows, key=lambda row: float(row["threshold"]))


def annotate_points(ax, rows):
    for row in rows:
        ax.annotate(
            f"t={float(row['threshold']):.2f}",
            (float(row["trigger_rate"]), float(row["scheduled_accuracy"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.85,
        )


def plot_trigger_vs_accuracy(summary_sets, labels, output_dir, title_prefix):
    fig, ax = plt.subplots(figsize=(8, 6))
    for rows, label in zip(summary_sets, labels):
        ordered = sort_rows(rows)
        x = [float(row["trigger_rate"]) for row in ordered]
        y = [float(row["scheduled_accuracy"]) for row in ordered]
        ax.plot(x, y, marker="o", label=label)
        annotate_points(ax, ordered)

    ax.set_xlabel("Trigger Rate")
    ax.set_ylabel("Scheduled Accuracy")
    ax.set_title(f"{title_prefix}: Trigger Rate vs Scheduled Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "trigger_rate_vs_scheduled_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_trigger_vs_gain(summary_sets, labels, output_dir, title_prefix):
    fig, ax = plt.subplots(figsize=(8, 6))
    for rows, label in zip(summary_sets, labels):
        ordered = sort_rows(rows)
        x = [float(row["trigger_rate"]) for row in ordered]
        y = [float(row["scheduled_gain_over_small"]) for row in ordered]
        ax.plot(x, y, marker="o", label=label)
        for row in ordered:
            ax.annotate(
                f"t={float(row['threshold']):.2f}",
                (float(row["trigger_rate"]), float(row["scheduled_gain_over_small"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                alpha=0.85,
            )

    ax.set_xlabel("Trigger Rate")
    ax.set_ylabel("Scheduled Gain Over Small")
    ax.set_title(f"{title_prefix}: Trigger Rate vs Gain Over Small")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "trigger_rate_vs_gain_over_small.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_threshold_curves(summary_sets, labels, output_dir, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for rows, label in zip(summary_sets, labels):
        ordered = sort_rows(rows)
        thresholds = [float(row["threshold"]) for row in ordered]
        accuracies = [float(row["scheduled_accuracy"]) for row in ordered]
        trigger_rates = [float(row["trigger_rate"]) for row in ordered]
        axes[0].plot(thresholds, accuracies, marker="o", label=label)
        axes[1].plot(thresholds, trigger_rates, marker="o", label=label)

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Scheduled Accuracy")
    axes[0].set_title(f"{title_prefix}: Threshold vs Accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Trigger Rate")
    axes[1].set_title(f"{title_prefix}: Threshold vs Trigger Rate")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    path = output_dir / "threshold_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    summary_paths = [Path(path) for path in args.summary_paths]
    labels = args.labels or [infer_label(path) for path in summary_paths]
    if len(labels) != len(summary_paths):
        raise ValueError("If --labels is provided, it must have the same length as --summary-paths.")

    summary_sets = [load_summary(path) for path in summary_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_trigger_vs_accuracy(summary_sets, labels, output_dir, args.title_prefix),
        plot_trigger_vs_gain(summary_sets, labels, output_dir, args.title_prefix),
        plot_threshold_curves(summary_sets, labels, output_dir, args.title_prefix),
    ]

    print("Saved plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
