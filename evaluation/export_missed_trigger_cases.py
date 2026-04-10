import argparse
import csv
import os

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FAILURE_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "scheduler_failure_analysis.csv")
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "dataset", "gsm8k_labeled_training_data_strict.pt")
DEFAULT_PROBE_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "probe_artifact.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "missed_trigger_case_details.csv")
DEFAULT_THRESHOLD = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description="Export detailed case sheets for missed_no_trigger scheduler failures.")
    parser.add_argument("--failure-analysis-path", default=DEFAULT_FAILURE_ANALYSIS_PATH, help="Path to the failure analysis CSV.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to the strict labeled chunk dataset.")
    parser.add_argument("--probe-artifact-path", default=DEFAULT_PROBE_ARTIFACT_PATH, help="Path to the probe artifact used by the scheduler.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Where to save the detailed missed-trigger CSV.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Scheduler threshold used in the analyzed run.")
    return parser.parse_args()


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def canonical_feature_name(name):
    aliases = {
        "boundary": "boundary_hidden_state",
        "mean": "mean_hidden_state",
        "entropy": "final_entropy",
        "top1_prob": "final_top1_prob",
        "margin": "final_margin",
    }
    return aliases.get(name, name)


def parse_feature_spec(feature_spec):
    parts = []
    for part in feature_spec.split("+"):
        token = part.strip()
        if token:
            parts.append(token)
    if not parts:
        raise ValueError(f"Empty feature spec: {feature_spec}")
    return parts


def build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec):
    values = []
    for token in parse_feature_spec(feature_spec):
        if token == "delta_prev":
            base = tensor_to_numpy(chunk["boundary_hidden_state"])
            if prev_chunk is None:
                value = np.zeros_like(base, dtype=np.float32)
            else:
                value = base - tensor_to_numpy(prev_chunk["boundary_hidden_state"])
        elif token == "abs_delta_prev":
            base = tensor_to_numpy(chunk["boundary_hidden_state"])
            if prev_chunk is None:
                value = np.zeros_like(base, dtype=np.float32)
            else:
                value = np.abs(base - tensor_to_numpy(prev_chunk["boundary_hidden_state"]))
        elif token == "relative_position":
            denom = max(total_chunks - 1, 1)
            value = np.asarray([int(chunk["chunk_id"]) / denom], dtype=np.float32)
        elif token == "remaining_ratio":
            denom = max(total_chunks - 1, 1)
            value = np.asarray([(denom - int(chunk["chunk_id"])) / denom], dtype=np.float32)
        else:
            feature_key = canonical_feature_name(token)
            if feature_key not in chunk:
                raise KeyError(f"Feature component '{token}' (resolved to '{feature_key}') not found in chunk.")
            value = tensor_to_numpy(chunk[feature_key])
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(values, axis=0)


def artifact_positive_score(artifact, positive_prob):
    label_key = artifact.get("label_key", "label")
    if label_key == "takeover_beneficial":
        return positive_prob
    return 1.0 - positive_prob


def load_failure_question_ids(path):
    question_ids = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("auto_bucket") == "missed_no_trigger":
                question_ids.append(int(row["question_id"]))
    return sorted(set(question_ids))


def group_labeled_rows(dataset):
    grouped = {}
    for row in dataset:
        question_id = int(row["question_id"])
        grouped.setdefault(question_id, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["chunk_id"]))
    return grouped


def score_question_chunks(chunks, artifact):
    scaler = artifact["scaler"]
    probe = artifact["probe"]
    feature_spec = artifact["feature_key"]
    total_chunks = len(chunks)
    rows = []
    for index, chunk in enumerate(chunks):
        prev_chunk = None if index == 0 else chunks[index - 1]
        rows.append(build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec))
    X = np.stack(rows)
    X_scaled = scaler.transform(X)
    positive_scores = probe.predict_proba(X_scaled)[:, 1]
    return [float(artifact_positive_score(artifact, score)) for score in positive_scores]


def export_cases(question_ids, grouped_rows, artifact, threshold, output_path):
    fieldnames = [
        "question_id",
        "question",
        "first_error_chunk_id",
        "first_error_chunk_text",
        "first_error_prefix_text",
        "first_error_label",
        "first_error_judge_error_type",
        "first_error_judge_reason",
        "first_error_score",
        "first_error_score_margin_to_threshold",
        "max_question_score",
        "max_question_score_chunk_id",
        "max_question_score_margin_to_threshold",
        "ground_truth_final_answer",
        "small_final_answer",
        "is_final_correct",
    ]

    rows = []
    for question_id in question_ids:
        chunks = grouped_rows.get(question_id)
        if not chunks:
            continue

        question_scores = score_question_chunks(chunks, artifact)
        first_error_index = None
        for index, chunk in enumerate(chunks):
            if int(chunk["label"]) == 0:
                first_error_index = index
                break
        if first_error_index is None:
            continue

        first_error_chunk = chunks[first_error_index]
        first_error_score = question_scores[first_error_index]
        max_score_index = int(np.argmax(np.asarray(question_scores, dtype=np.float32)))
        max_score_chunk = chunks[max_score_index]
        max_score_value = question_scores[max_score_index]

        rows.append(
            {
                "question_id": question_id,
                "question": first_error_chunk.get("question", ""),
                "first_error_chunk_id": int(first_error_chunk["chunk_id"]),
                "first_error_chunk_text": first_error_chunk.get("chunk_text", ""),
                "first_error_prefix_text": first_error_chunk.get("prefix_text", ""),
                "first_error_label": int(first_error_chunk["label"]),
                "first_error_judge_error_type": first_error_chunk.get("judge_error_type", ""),
                "first_error_judge_reason": first_error_chunk.get("judge_reason", ""),
                "first_error_score": first_error_score,
                "first_error_score_margin_to_threshold": first_error_score - threshold,
                "max_question_score": max_score_value,
                "max_question_score_chunk_id": int(max_score_chunk["chunk_id"]),
                "max_question_score_margin_to_threshold": max_score_value - threshold,
                "ground_truth_final_answer": first_error_chunk.get("ground_truth_final_answer", ""),
                "small_final_answer": first_error_chunk.get("model_final_answer", ""),
                "is_final_correct": first_error_chunk.get("is_final_correct", ""),
            }
        )

    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} missed-trigger case rows to: {output_path}")


def main():
    args = parse_args()
    question_ids = load_failure_question_ids(args.failure_analysis_path)
    print(f"Missed-no-trigger questions found: {len(question_ids)}")

    dataset = torch.load(args.label_path)
    artifact = torch.load(args.probe_artifact_path)
    grouped_rows = group_labeled_rows(dataset)

    export_cases(question_ids, grouped_rows, artifact, args.threshold, args.output_path)


if __name__ == "__main__":
    main()

