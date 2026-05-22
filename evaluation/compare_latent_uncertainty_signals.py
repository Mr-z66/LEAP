import argparse
import csv
import json
import math
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core_package.config import PROBE_TRAIN
from core_package.probes.train_probe_artifact_torch import build_feature_vector, build_question_records


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs", "latent_signal_comparison")


RAW_SIGNALS = {
    "entropy": "final_entropy",
    "neg_top1_prob": "final_top1_prob",
    "neg_margin": "final_margin",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare final-layer latent features against token-uncertainty signals for chunk error detection."
    )
    parser.add_argument("--label-path", default=PROBE_TRAIN.label_path, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default=PROBE_TRAIN.label_key, help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument(
        "--feature-specs",
        default="boundary,mean,boundary+mean",
        help="Comma-separated feature specs to train probes on.",
    )
    parser.add_argument(
        "--raw-signals",
        default="entropy,neg_top1_prob,neg_margin",
        help="Comma-separated raw uncertainty signals to evaluate without training.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "mlp"],
        default="logreg",
        help="Classifier used for learned probes.",
    )
    parser.add_argument("--test-size", type=float, default=PROBE_TRAIN.test_size, help="Held-out question ratio.")
    parser.add_argument("--random-state", type=int, default=PROBE_TRAIN.random_state, help="Question split seed.")
    parser.add_argument("--mlp-hidden-layers", default="128,32", help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--mlp-max-iter", type=int, default=200, help="Max MLP iterations.")
    parser.add_argument(
        "--trigger-rates",
        default="0.05,0.10,0.20",
        help="Comma-separated top-score trigger rates for recall/precision reporting.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV/JSON/Markdown.")
    parser.add_argument("--output-prefix", default=None, help="Optional output filename prefix.")
    return parser.parse_args()


def parse_csv_text(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_csv_floats(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_hidden_layers(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("--mlp-hidden-layers cannot be empty.")
    return tuple(values)


def scalar_value(chunk, key):
    value = chunk.get(key)
    if value is None:
        raise KeyError(f"Missing raw signal field: {key}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0])


def build_rows(question_records, label_key):
    rows = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            label = int(chunk[label_key])
            if label not in {0, 1}:
                continue
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(
                {
                    "question_id": int(question_id),
                    "chunk": chunk,
                    "prev_chunk": prev_chunk,
                    "total_chunks": total_chunks,
                    "error_label": 1 - label,
                }
            )
    if not rows:
        raise ValueError(f"No valid rows found for label_key={label_key!r}.")
    return rows


def split_rows(rows, test_size, random_state):
    groups = np.asarray([row["question_id"] for row in rows], dtype=np.int64)
    labels = np.asarray([row["error_label"] for row in rows], dtype=np.int64)
    indices = np.arange(len(rows))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(indices, labels, groups=groups))
    return train_idx, test_idx


def recall_precision_at_trigger_rate(y_true, scores, trigger_rate):
    if not 0.0 < trigger_rate <= 1.0:
        raise ValueError(f"trigger_rate must be in (0, 1]: {trigger_rate}")
    k = max(1, int(math.ceil(trigger_rate * len(scores))))
    order = np.argsort(scores)[::-1]
    selected = np.zeros_like(y_true, dtype=bool)
    selected[order[:k]] = True
    positives = y_true == 1
    true_selected = selected & positives
    recall = true_selected.sum() / max(positives.sum(), 1)
    precision = true_selected.sum() / max(selected.sum(), 1)
    return float(recall), float(precision)


def metric_summary(y_true, scores, trigger_rates):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    summary = {
        "error_roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "error_pr_auc": float(average_precision_score(y_true, scores)),
        "best_error_f1": float(f1[best_idx]),
        "best_error_precision": float(precision[best_idx]),
        "best_error_recall": float(recall[best_idx]),
        "best_threshold": float(thresholds[best_idx]) if best_idx < len(thresholds) else float("inf"),
    }
    for trigger_rate in trigger_rates:
        recall_at_k, precision_at_k = recall_precision_at_trigger_rate(y_true, scores, trigger_rate)
        suffix = str(trigger_rate).replace(".", "p")
        summary[f"recall_at_trigger_{suffix}"] = recall_at_k
        summary[f"precision_at_trigger_{suffix}"] = precision_at_k
    return summary


def make_classifier(args):
    if args.classifier == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    else:
        model = MLPClassifier(
            hidden_layer_sizes=parse_hidden_layers(args.mlp_hidden_layers),
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            random_state=args.random_state,
        )
    return make_pipeline(StandardScaler(), model)


def feature_matrix(rows, feature_spec):
    values = [
        build_feature_vector(row["chunk"], row["prev_chunk"], row["total_chunks"], feature_spec)
        for row in rows
    ]
    return np.stack(values).astype(np.float32)


def raw_signal_scores(rows, signal_name):
    key = RAW_SIGNALS[signal_name]
    values = np.asarray([scalar_value(row["chunk"], key) for row in rows], dtype=np.float64)
    if signal_name in {"neg_top1_prob", "neg_margin"}:
        values = -values
    return values


def evaluate_raw_signal(rows, test_idx, signal_name, trigger_rates):
    y_test = np.asarray([rows[i]["error_label"] for i in test_idx], dtype=np.int64)
    scores = raw_signal_scores([rows[i] for i in test_idx], signal_name)
    summary = metric_summary(y_test, scores, trigger_rates)
    return {
        "signal": signal_name,
        "feature_spec": RAW_SIGNALS[signal_name],
        "model": "raw_score",
        "feature_dim": 1,
        **summary,
    }


def evaluate_learned_probe(rows, train_idx, test_idx, feature_spec, args, trigger_rates):
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    X_train = feature_matrix(train_rows, feature_spec)
    X_test = feature_matrix(test_rows, feature_spec)
    y_train = np.asarray([row["error_label"] for row in train_rows], dtype=np.int64)
    y_test = np.asarray([row["error_label"] for row in test_rows], dtype=np.int64)

    model = make_classifier(args)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    summary = metric_summary(y_test, scores, trigger_rates)
    return {
        "signal": f"{feature_spec}_{args.classifier}",
        "feature_spec": feature_spec,
        "model": args.classifier,
        "feature_dim": int(X_train.shape[1]),
        **summary,
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, trigger_rates):
    core_columns = [
        "signal",
        "model",
        "feature_dim",
        "error_roc_auc",
        "error_pr_auc",
        "best_error_f1",
    ]
    trigger_columns = []
    for trigger_rate in trigger_rates:
        suffix = str(trigger_rate).replace(".", "p")
        trigger_columns.extend([f"recall_at_trigger_{suffix}", f"precision_at_trigger_{suffix}"])
    columns = core_columns + trigger_columns
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            cells = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            handle.write("| " + " | ".join(cells) + " |\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = torch.load(args.label_path, weights_only=False)
    question_records = build_question_records(dataset, args.label_key)
    rows = build_rows(question_records, args.label_key)
    train_idx, test_idx = split_rows(rows, args.test_size, args.random_state)
    trigger_rates = parse_csv_floats(args.trigger_rates)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(args.label_path))[0]

    results = []
    for signal_name in parse_csv_text(args.raw_signals):
        if signal_name not in RAW_SIGNALS:
            available = ", ".join(sorted(RAW_SIGNALS))
            raise ValueError(f"Unknown raw signal {signal_name!r}. Available: {available}")
        results.append(evaluate_raw_signal(rows, test_idx, signal_name, trigger_rates))

    for feature_spec in parse_csv_text(args.feature_specs):
        results.append(evaluate_learned_probe(rows, train_idx, test_idx, feature_spec, args, trigger_rates))

    metadata = {
        "label_path": args.label_path,
        "label_key": args.label_key,
        "classifier": args.classifier,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "train_chunks": int(len(train_idx)),
        "test_chunks": int(len(test_idx)),
        "train_questions": int(len(set(rows[i]["question_id"] for i in train_idx))),
        "test_questions": int(len(set(rows[i]["question_id"] for i in test_idx))),
        "trigger_rates": trigger_rates,
    }
    for row in results:
        row.update(metadata)

    csv_path = os.path.join(args.output_dir, f"{output_prefix}_latent_vs_uncertainty.csv")
    json_path = os.path.join(args.output_dir, f"{output_prefix}_latent_vs_uncertainty.json")
    md_path = os.path.join(args.output_dir, f"{output_prefix}_latent_vs_uncertainty.md")

    write_csv(csv_path, results)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": results}, handle, indent=2, ensure_ascii=True)
    write_markdown(md_path, results, trigger_rates)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print("\nSignal comparison")
    print("=" * 80)
    for row in results:
        print(
            f"{row['signal']:<24} "
            f"ROC-AUC={row['error_roc_auc']:.4f} "
            f"PR-AUC={row['error_pr_auc']:.4f} "
            f"F1={row['best_error_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

