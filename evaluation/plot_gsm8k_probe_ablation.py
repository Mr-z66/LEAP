import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


INPUT_PATHS = [
    "result/analysis_outputs/gsm8k_hidden_only_chunk_scores.csv",
    "result/analysis_outputs/gsm8k_entropy_only_chunk_scores.csv",
]
MERGED_OUTPUT_PATH = "result/analysis_outputs/gsm8k_probe_ablation_chunk_scores.csv"
DISTRIBUTION_FIGURE_PATH = "result/analysis_outputs/gsm8k_probe_score_distribution.png"
ROC_FIGURE_PATH = "result/analysis_outputs/gsm8k_probe_roc.png"


def load_scores():
    frames = [pd.read_csv(path) for path in INPUT_PATHS]
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
    df.to_csv(MERGED_OUTPUT_PATH, index=False)
    return df


def plot_distribution(df):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    plot_specs = [
        ("hidden_only", "Hidden-only"),
        ("entropy_only", "Entropy-only"),
    ]
    for ax, feature_set, title in [(axes[0], *plot_specs[0]), (axes[1], *plot_specs[1])]:
        sub = df[df["feature_set"] == feature_set].copy()
        sub["label_name"] = sub["label"].map({1: "Correct chunk", 0: "Error chunk"})
        sns.kdeplot(
            data=sub,
            x="positive_prob",
            hue="label_name",
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=2,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Probe positive probability")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(DISTRIBUTION_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(df):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5, 5))

    for feature_set, label_name in [
        ("hidden_only", "Hidden-only"),
        ("entropy_only", "Entropy-only"),
    ]:
        sub = df[df["feature_set"] == feature_set]
        fpr, tpr, _ = roc_curve(sub["label"], sub["positive_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("GSM8K Chunk Classification ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_scores()
    print(f"Saved merged CSV to: {MERGED_OUTPUT_PATH}")
    print(df.groupby("feature_set").size())
    plot_distribution(df)
    print(f"Saved distribution figure to: {DISTRIBUTION_FIGURE_PATH}")
    plot_roc(df)
    print(f"Saved ROC figure to: {ROC_FIGURE_PATH}")


if __name__ == "__main__":
    main()
