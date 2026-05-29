import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs", "preliminary_figures")
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "dataset", "mixed_probe_labels_fallback_second_pass", "gsm8k_calib_labels.pt")


RAW_SCORE_FIELDS = {
    "entropy": ("final_entropy", -1.0, "Entropy"),
    "mean_entropy": ("mean_entropy", -1.0, "Mean entropy"),
    "max_entropy": ("max_entropy", -1.0, "Max entropy"),
    "top1": ("final_top1_prob", 1.0),
    "mean_top1": ("mean_top1_prob", 1.0, "Mean top-1 probability"),
    "min_top1": ("min_top1_prob", 1.0, "Minimum top-1 probability"),
    "margin": ("final_margin", 1.0),
    "mean_margin": ("mean_margin", 1.0, "Mean logit margin"),
    "min_margin": ("min_margin", 1.0, "Minimum logit margin"),
}


PANEL_TITLES = {
    "entropy": "Entropy confidence",
    "mean_entropy": "Mean entropy conf.",
    "max_entropy": "Max entropy conf.",
    "top1": "Top-1 probability",
    "mean_top1": "Mean top-1 prob.",
    "min_top1": "Min top-1 prob.",
    "margin": "Logit margin",
    "mean_margin": "Mean margin",
    "min_margin": "Min margin",
    "boundary": "Boundary probe",
    "mean": "Mean probe",
    "boundary+mean": "Boundary+mean probe",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot preliminary score distributions for correct vs wrong chunks."
    )
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument(
        "--panels",
        default="entropy,top1,margin,boundary,mean,boundary+mean",
        help="Comma-separated panels. Raw: entropy,top1,margin. Probe: boundary,mean,boundary+mean.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "mlp"],
        default="logreg",
        help="Classifier used for learned probe panels.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out question ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Question split seed.")
    parser.add_argument("--mlp-hidden-layers", default="128,32", help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--mlp-max-iter", type=int, default=200, help="Max MLP iterations.")
    parser.add_argument("--bins", type=int, default=18, help="Histogram bin count.")
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="Plot raw surface signal values instead of normalized 0-100 confidence scores.",
    )
    parser.add_argument(
        "--xlim",
        default="0,100",
        help="Comma-separated x-axis limits. Use raw units with --raw-scale, e.g. 0,2 for entropy.",
    )
    parser.add_argument("--density", action="store_true", help="Plot normalized density instead of raw counts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-prefix", default=None, help="Optional output filename prefix.")
    return parser.parse_args()


def parse_xlim(text):
    pieces = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(pieces) != 2 or pieces[0] >= pieces[1]:
        raise ValueError("--xlim must be two increasing numbers, e.g. 70,100")
    return pieces[0], pieces[1]


def scalar_value(chunk, key):
    import torch

    value = chunk.get(key)
    if value is None:
        raise KeyError(f"Missing signal field: {key}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def minmax_percent(values):
    values = np.asarray(values, dtype=np.float64)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return np.full_like(values, 50.0, dtype=np.float64)
    return 100.0 * (values - low) / (high - low)


def raw_panel_scores(rows, indices, panel, raw_scale):
    key, direction, _ = RAW_SCORE_FIELDS[panel]
    values = np.asarray([scalar_value(rows[i]["chunk"], key) for i in indices], dtype=np.float64)
    if raw_scale:
        return values
    return minmax_percent(direction * values)


def probe_correctness_scores(rows, train_idx, test_idx, feature_spec, args):
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    x_train = feature_matrix(train_rows, feature_spec)
    x_test = feature_matrix(test_rows, feature_spec)
    y_train_error = np.asarray([row["error_label"] for row in train_rows], dtype=np.int64)

    model = make_classifier(args)
    model.fit(x_train, y_train_error)
    error_scores = model.predict_proba(x_test)[:, 1]
    return 100.0 * (1.0 - error_scores)


def collect_panel_scores(rows, train_idx, test_idx, panel, args):
    if panel in RAW_SCORE_FIELDS:
        return raw_panel_scores(rows, test_idx, panel, args.raw_scale)
    return probe_correctness_scores(rows, train_idx, test_idx, panel, args)


def plot_distributions(panel_rows, output_png, output_pdf, bins, xlim, density):
    colors = {
        "all": "#8E7CC3",
        "correct": "#2F6DA3",
        "wrong": "#D94136",
    }
    fig, axes = plt.subplots(
        1,
        len(panel_rows),
        figsize=(2.15 * len(panel_rows), 2.35),
        sharey=True,
        constrained_layout=False,
    )
    if len(panel_rows) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panel_rows):
        scores = panel["scores"]
        correct_scores = scores[panel["labels"] == 1]
        wrong_scores = scores[panel["labels"] == 0]
        hist_range = xlim

        ax.hist(scores, bins=bins, range=hist_range, density=density, color=colors["all"], alpha=0.38, label="all")
        ax.hist(
            correct_scores,
            bins=bins,
            range=hist_range,
            density=density,
            color=colors["correct"],
            alpha=0.58,
            label="correct",
        )
        ax.hist(
            wrong_scores,
            bins=bins,
            range=hist_range,
            density=density,
            color=colors["wrong"],
            alpha=0.58,
            label="wrong",
        )

        correct_mean = float(np.mean(correct_scores)) if len(correct_scores) else float("nan")
        wrong_mean = float(np.mean(wrong_scores)) if len(wrong_scores) else float("nan")
        ax.axvline(correct_mean, color=colors["correct"], linewidth=1.5)
        ax.axvline(wrong_mean, color=colors["wrong"], linewidth=1.5)
        ax.text(correct_mean, 0.97, f"{correct_mean:.1f}", transform=ax.get_xaxis_transform(), color=colors["correct"],
                fontsize=8, fontweight="bold", ha="center", va="top")
        ax.text(wrong_mean, 0.88, f"{wrong_mean:.1f}", transform=ax.get_xaxis_transform(), color=colors["wrong"],
                fontsize=8, fontweight="bold", ha="center", va="top")

        ax.set_xlim(*xlim)
        ax.set_title(panel["title"], fontsize=9, pad=7)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.tick_params(axis="both", labelsize=8, length=2)
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.03, 1.03), fontsize=8)
    xlabel = panel_rows[0].get("xlabel", "Correctness score (0-100, higher is more likely correct)")
    fig.supxlabel(xlabel, y=0.06, fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.27, wspace=0.08)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    import torch

    global build_rows, feature_matrix, make_classifier, split_rows
    from core_package.probes.train_probe_artifact_torch import build_question_records
    from evaluation.compare_latent_uncertainty_signals import (
        build_rows,
        feature_matrix,
        make_classifier,
        split_rows,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    panels = [part.strip() for part in args.panels.split(",") if part.strip()]

    dataset = torch.load(args.label_path, weights_only=False)
    question_records = build_question_records(dataset, args.label_key)
    rows = build_rows(question_records, args.label_key)
    train_idx, test_idx = split_rows(rows, args.test_size, args.random_state)
    labels = np.asarray([1 - rows[i]["error_label"] for i in test_idx], dtype=np.int64)

    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]
    panel_rows = []
    for panel in panels:
        if panel not in RAW_SCORE_FIELDS and panel not in {"boundary", "mean", "boundary+mean"}:
            raise ValueError(f"Unsupported panel: {panel}")
        panel_rows.append(
            {
                "name": panel,
                "title": PANEL_TITLES.get(panel, panel),
                "scores": collect_panel_scores(rows, train_idx, test_idx, panel, args),
                "labels": labels,
                "xlabel": RAW_SCORE_FIELDS[panel][2] if args.raw_scale and panel in RAW_SCORE_FIELDS else "Correctness score (0-100, higher is more likely correct)",
            }
        )

    output_png = os.path.join(args.output_dir, f"{output_prefix}_score_distributions.png")
    output_pdf = os.path.join(args.output_dir, f"{output_prefix}_score_distributions.pdf")
    plot_distributions(panel_rows, output_png, output_pdf, args.bins, parse_xlim(args.xlim), args.density)

    print(f"Wrote: {output_png}")
    print(f"Wrote: {output_pdf}")
    print(f"Test chunks: {len(test_idx)}")
    print(f"Correct chunks: {int(labels.sum())}")
    print(f"Wrong chunks: {int((labels == 0).sum())}")


if __name__ == "__main__":
    main()
