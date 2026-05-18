import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from core_package.probes.train_probe_artifact_torch import (
    TorchMLPProbe,
    build_feature_arrays,
    build_question_records,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze offline next-chunk risk probe predictions."
    )
    parser.add_argument("--label-path", required=True, help="Path to the relabeled .pt dataset.")
    parser.add_argument("--artifact-path", required=True, help="Path to the trained probe artifact.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive probability threshold.")
    parser.add_argument("--show-questions", type=int, default=10, help="Top risky questions to print.")
    return parser.parse_args()


def load_probe(artifact):
    probe = artifact.get("probe")
    if probe is not None:
        probe.eval()
        return probe

    hidden_layers = tuple(int(x) for x in artifact["config"]["hidden_layers"].split(",") if x.strip())
    probe = TorchMLPProbe(
        input_dim=int(artifact["feature_dim"]),
        hidden_layers=hidden_layers,
        dropout=float(artifact["config"].get("dropout", 0.0)),
    )
    probe.load_state_dict(artifact["probe_state_dict"])
    probe.eval()
    return probe


def rows_from_dataset(dataset, feature_key):
    question_records = build_question_records(dataset, "label")
    X, y, groups, _sample_weights = build_feature_arrays(
        question_records,
        feature_key,
        "label",
        argparse.Namespace(
            low_entropy_error_final_entropy_max=None,
            low_entropy_error_final_top1_min=None,
            low_entropy_error_weight=1.0,
        ),
    )

    rows = []
    cursor = 0
    for question_id, record in question_records.items():
        for chunk in record["chunks"]:
            rows.append(
                {
                    "question_id": int(question_id),
                    "chunk_id": int(chunk["chunk_id"]),
                    "raw_label": int(chunk["label"]),
                    "current_label": int(chunk.get("current_label", -1)),
                    "next1_error": int(chunk.get("next1_error", -1)),
                    "nextk_error": int(chunk.get("nextk_error", -1)),
                    "first_future_error_offset": chunk.get("first_future_error_offset"),
                    "features": X[cursor],
                }
            )
            cursor += 1
    return rows


def main():
    args = parse_args()
    label_path = Path(args.label_path)
    artifact_path = Path(args.artifact_path)

    dataset = torch.load(label_path, weights_only=False)
    artifact = torch.load(artifact_path, weights_only=False)
    probe = load_probe(artifact)
    scaler = artifact["scaler"]
    feature_key = artifact["feature_key"]
    test_question_ids = set(int(qid) for qid in artifact["test_question_ids"])

    all_rows = rows_from_dataset(dataset, feature_key)
    test_rows = [row for row in all_rows if row["question_id"] in test_question_ids]
    if not test_rows:
        raise ValueError("No rows matched the artifact's held-out test_question_ids.")

    X_test = np.stack([row["features"] for row in test_rows]).astype(np.float32)
    y_test = np.asarray([row["raw_label"] for row in test_rows], dtype=np.int64)
    X_test_scaled = scaler.transform(X_test)
    pos_probs = probe.predict_proba(X_test_scaled)[:, 1]
    pred_labels = (pos_probs >= args.threshold).astype(np.int64)

    print(f"Artifact: {artifact_path}")
    print(f"Label data: {label_path}")
    print(f"Feature key: {feature_key}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Test questions: {len(test_question_ids)} | Test rows: {len(test_rows)}")
    print(f"True label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"Pred label distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")
    print(f"Predicted risk rate: {pred_labels.mean():.4f}")
    print(f"Mean predicted risk prob: {pos_probs.mean():.4f}")
    print("Confusion matrix [true 0/1 x pred 0/1]:")
    print(confusion_matrix(y_test, pred_labels, labels=[0, 1]).tolist())
    print(classification_report(y_test, pred_labels, digits=4, zero_division=0))

    bucket_stats = {}
    for row, prob, pred in zip(test_rows, pos_probs, pred_labels):
        bucket = int(row["first_future_error_offset"] or 99)
        stats = bucket_stats.setdefault(bucket, {"count": 0, "prob_sum": 0.0, "pred_sum": 0})
        stats["count"] += 1
        stats["prob_sum"] += float(prob)
        stats["pred_sum"] += int(pred)

    print("First future error offset summary (99 means no error inside horizon):")
    for bucket in sorted(bucket_stats):
        stats = bucket_stats[bucket]
        print(
            f"  offset={bucket}: count={stats['count']} | "
            f"mean_prob={stats['prob_sum'] / stats['count']:.4f} | "
            f"pred_risk_rate={stats['pred_sum'] / stats['count']:.4f}"
        )

    by_question = {}
    for row, prob, pred in zip(test_rows, pos_probs, pred_labels):
        question_stats = by_question.setdefault(
            row["question_id"],
            {"count": 0, "prob_sum": 0.0, "pred_count": 0, "max_prob": 0.0},
        )
        question_stats["count"] += 1
        question_stats["prob_sum"] += float(prob)
        question_stats["pred_count"] += int(pred)
        question_stats["max_prob"] = max(question_stats["max_prob"], float(prob))

    ranked_questions = sorted(
        by_question.items(),
        key=lambda item: (item[1]["max_prob"], item[1]["prob_sum"] / item[1]["count"]),
        reverse=True,
    )
    print(f"Top {min(args.show_questions, len(ranked_questions))} questions by max risk probability:")
    for question_id, stats in ranked_questions[: args.show_questions]:
        print(
            f"  qid={question_id} | rows={stats['count']} | "
            f"mean_prob={stats['prob_sum'] / stats['count']:.4f} | "
            f"max_prob={stats['max_prob']:.4f} | "
            f"pred_risk_rows={stats['pred_count']}"
        )


if __name__ == "__main__":
    main()
