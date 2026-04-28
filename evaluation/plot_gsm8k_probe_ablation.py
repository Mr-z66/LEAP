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
DISTRIBUTION_FIGURE_PATH = "result/analysis_outputs/gsm8k_error_score_distribution.png"
BOX_FIGURE_PATH = "result/analysis_outputs/gsm8k_error_score_boxplot.png"
ROC_FIGURE_PATH = "result/analysis_outputs/gsm8k_error_detection_roc.png"


def load_scores():
    frames = [pd.read_csv(path) for path in INPUT_PATHS]
    df = pd.concat(frames, ignore_index=True)
    df["label_name"] = df["label"].map({1: "Correct chunk", 0: "Error chunk"})
    df["is_error"] = (df["label"] == 0).astype(int)
    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
    df.to_csv(MERGED_OUTPUT_PATH, index=False)
    return df


def plot_distribution(df):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    palette = {"Correct chunk": "#4C78A8", "Error chunk": "#E45756"}

    plot_specs = [
        ("hidden_only", "Hidden-only"),
        ("entropy_only", "Entropy-only"),
    ]
    for ax, feature_set, title in [(axes[0], *plot_specs[0]), (axes[1], *plot_specs[1])]:
        sub = df[df["feature_set"] == feature_set].copy()
        sns.kdeplot(
            data=sub,
            x="trigger_score",
            hue="label_name",
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=2,
            palette=palette,
            ax=ax,
        )
        x_max = max(0.05, float(sub["trigger_score"].quantile(0.995)))
        ax.set_xlim(0.0, x_max)
        ax.set_title(title)
        ax.set_xlabel("Error score (1 - correct probability)")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(DISTRIBUTION_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_box(df):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    palette = {"Correct chunk": "#4C78A8", "Error chunk": "#E45756"}

    sns.boxplot(
        data=df,
        x="feature_set",
        y="trigger_score",
        hue="label_name",
        palette=palette,
        ax=ax,
        showfliers=False,
    )
    ax.set_xticklabels(["Hidden-only", "Entropy-only"])
    ax.set_xlabel("")
    ax.set_ylabel("Error score (1 - correct probability)")
    ax.set_title("GSM8K Error-Score Separation")
    plt.tight_layout()
    plt.savefig(BOX_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc(df):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(6, 6))

    for feature_set, label_name, color in [
        ("hidden_only", "Hidden-only", "#4C78A8"),
        ("entropy_only", "Entropy-only", "#F58518"),
    ]:
        sub = df[df["feature_set"] == feature_set]
        fpr, tpr, _ = roc_curve(sub["is_error"], sub["trigger_score"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, color=color, label=f"{label_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("GSM8K ROC for Error-Chunk Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_scores()
    print(f"Saved merged CSV to: {MERGED_OUTPUT_PATH}")
    print(df.groupby(['feature_set', 'label']).size())
    print("\nMean error score by feature_set,label")
    print(df.groupby(["feature_set", "label"])["trigger_score"].mean())
    plot_distribution(df)
    print(f"Saved distribution figure to: {DISTRIBUTION_FIGURE_PATH}")
    plot_box(df)
    print(f"Saved box figure to: {BOX_FIGURE_PATH}")
    plot_roc(df)
    print(f"Saved ROC figure to: {ROC_FIGURE_PATH}")


if __name__ == "__main__":
    main()
