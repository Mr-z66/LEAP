import argparse
import os
import re

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "probe_artifact_torch.pt")
DEFAULT_THRESHOLD_GRID = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


class TorchMLPProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        dims = [input_dim, *hidden_layers, 1]
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.from_numpy(X).to(torch.float32)
            else:
                X_tensor = X.to(torch.float32)
            logits = self.forward(X_tensor)
            pos_prob = torch.sigmoid(logits).cpu().numpy()
        neg_prob = 1.0 - pos_prob
        return np.stack([neg_prob, pos_prob], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PyTorch probe artifact on its held-out strict split.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the strict labeled chunk dataset.")
    parser.add_argument("--artifact-path", default=DEFAULT_ARTIFACT_PATH, help="Path to the saved PyTorch probe artifact.")
    parser.add_argument("--threshold-grid", default=DEFAULT_THRESHOLD_GRID, help="Comma-separated thresholds for scanning error-score decision points.")
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


DERIVED_SCALAR_FEATURES = {
    "relative_position",
    "remaining_ratio",
    "digit_count",
    "operator_count",
    "numeric_density",
    "contains_multiple_numbers",
    "has_equation_like_pattern",
    "has_finalization_cue",
    "entropy_delta",
    "margin_delta",
    "top1_prob_delta",
    "boundary_cosine_drift",
    "boundary_l2_drift",
}


def chunk_text(chunk):
    return str(chunk.get("chunk_text", "") or "")


def chunk_scalar_feature(chunk, token):
    text = chunk_text(chunk)
    compact_text = text.replace(",", "")
    digits = re.findall(r"\d", compact_text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", compact_text)
    non_space_chars = len(re.findall(r"\S", text))
    lower_text = text.lower()

    if token == "digit_count":
        return np.asarray([len(digits)], dtype=np.float32)
    if token == "operator_count":
        return np.asarray([sum(text.count(symbol) for symbol in "+-*/=")], dtype=np.float32)
    if token == "numeric_density":
        return np.asarray([len(digits) / max(non_space_chars, 1)], dtype=np.float32)
    if token == "contains_multiple_numbers":
        return np.asarray([1.0 if len(set(numbers)) >= 2 else 0.0], dtype=np.float32)
    if token == "has_equation_like_pattern":
        has_pattern = "=" in text or bool(re.search(r"\d\s*[-+*/]\s*\d", compact_text))
        return np.asarray([1.0 if has_pattern else 0.0], dtype=np.float32)
    if token == "has_finalization_cue":
        has_cue = bool(re.search(r"\b(therefore|thus|so|answer|final answer|total)\b", lower_text))
        return np.asarray([1.0 if has_cue else 0.0], dtype=np.float32)
    raise KeyError(f"Unsupported derived scalar feature: {token}")


def chunk_confidence_scalar(chunk, token):
    if token == "entropy_delta":
        key = "final_entropy"
    elif token == "margin_delta":
        key = "final_margin"
    elif token == "top1_prob_delta":
        key = "final_top1_prob"
    else:
        raise KeyError(f"Unsupported confidence delta token: {token}")
    return float(tensor_to_numpy(chunk[key]).reshape(-1)[0])


def boundary_hidden_drift(chunk, prev_chunk, token):
    if prev_chunk is None:
        return np.asarray([0.0], dtype=np.float32)
    current = tensor_to_numpy(chunk["boundary_hidden_state"]).reshape(-1)
    previous = tensor_to_numpy(prev_chunk["boundary_hidden_state"]).reshape(-1)
    if token == "boundary_l2_drift":
        return np.asarray([float(np.linalg.norm(current - previous))], dtype=np.float32)
    if token == "boundary_cosine_drift":
        denom = float(np.linalg.norm(current) * np.linalg.norm(previous))
        if denom <= 1e-12:
            return np.asarray([0.0], dtype=np.float32)
        cosine = float(np.dot(current, previous) / denom)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return np.asarray([1.0 - cosine], dtype=np.float32)
    raise KeyError(f"Unsupported boundary drift token: {token}")


def build_question_records(dataset):
    question_records = {}
    for item in dataset:
        if int(item["label"]) not in {0, 1}:
            continue
        question_id = int(item["question_id"])
        record = question_records.setdefault(question_id, {"chunks": []})
        record["chunks"].append(item)

    for record in question_records.values():
        record["chunks"] = sorted(record["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    return question_records


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
        elif token in {"entropy_delta", "margin_delta", "top1_prob_delta"}:
            current = chunk_confidence_scalar(chunk, token)
            previous = 0.0 if prev_chunk is None else chunk_confidence_scalar(prev_chunk, token)
            value = np.asarray([current - previous], dtype=np.float32)
        elif token in {"boundary_cosine_drift", "boundary_l2_drift"}:
            value = boundary_hidden_drift(chunk, prev_chunk, token)
        elif token in DERIVED_SCALAR_FEATURES:
            value = chunk_scalar_feature(chunk, token)
        else:
            feature_key = canonical_feature_name(token)
            if feature_key not in chunk:
                raise KeyError(f"Feature component '{token}' (resolved to '{feature_key}') not found in chunk.")
            value = tensor_to_numpy(chunk[feature_key])
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(values, axis=0)


def parse_csv_floats(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric threshold.")
    return values


def main():
    args = parse_args()

    print(f"Loading labeled chunk dataset from: {args.data_path}")
    dataset = torch.load(args.data_path, weights_only=False)
    question_records = build_question_records(dataset)

    print(f"Loading PyTorch probe artifact from: {args.artifact_path}")
    artifact = torch.load(args.artifact_path, weights_only=False)
    feature_key = artifact["feature_key"]
    test_question_ids = set(int(qid) for qid in artifact["test_question_ids"])
    scaler = artifact["scaler"]
    probe = artifact.get("probe")
    if probe is None:
        hidden_layers = tuple(int(x) for x in artifact["config"]["hidden_layers"].split(",") if x.strip())
        probe = TorchMLPProbe(input_dim=int(artifact["feature_dim"]), hidden_layers=hidden_layers)
        probe.load_state_dict(artifact["probe_state_dict"])
    probe.eval()

    rows = []
    for question_id in sorted(test_question_ids):
        record = question_records[question_id]
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append({
                "question_id": question_id,
                "label": int(chunk["label"]),
                "features": build_feature_vector(chunk, prev_chunk, total_chunks, feature_key),
            })

    X_test = np.stack([row["features"] for row in rows]).astype(np.float32)
    y_test = np.asarray([row["label"] for row in rows], dtype=np.int64)
    X_test_scaled = scaler.transform(X_test)

    positive_probs = probe.predict_proba(X_test_scaled)[:, 1]
    error_scores = 1.0 - positive_probs
    predicted_labels = (positive_probs >= 0.5).astype(np.int64)

    prefix_correct_auroc = roc_auc_score(y_test, positive_probs)
    prefix_correct_auprc = average_precision_score(y_test, positive_probs)
    error_auprc = average_precision_score(1 - y_test, error_scores)
    balanced_accuracy = balanced_accuracy_score(y_test, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predicted_labels,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    print("\nHeld-out strict evaluation (PyTorch probe)")
    print("=" * 50)
    print(f"Feature: {feature_key}")
    print(f"Chunks: {len(rows)} | Questions: {len(test_question_ids)}")
    print(f"Prefix-correct AUROC: {prefix_correct_auroc:.4f}")
    print(f"Prefix-correct AUPRC: {prefix_correct_auprc:.4f}")
    print(f"Error-class AUPRC: {error_auprc:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Error precision: {precision[0]:.4f}")
    print(f"Error recall: {recall[0]:.4f}")
    print(f"Error F1: {f1[0]:.4f}")
    print(f"Confusion matrix [true 0/1 x pred 0/1]: {confusion_matrix(y_test, predicted_labels, labels=[0, 1]).tolist()}")
    print(classification_report(y_test, predicted_labels, digits=4, zero_division=0))

    print("\nThreshold scan on held-out split")
    print("-" * 50)
    best_threshold = None
    best_result = None
    for threshold in parse_csv_floats(args.threshold_grid):
        threshold_predictions = (error_scores >= threshold).astype(np.int64)
        threshold_positive_predictions = 1 - threshold_predictions
        threshold_balanced_accuracy = balanced_accuracy_score(y_test, threshold_positive_predictions)
        threshold_precision, threshold_recall, threshold_f1, _ = precision_recall_fscore_support(
            y_test,
            threshold_positive_predictions,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        current = {
            "threshold": threshold,
            "error_precision": float(threshold_precision[0]),
            "error_recall": float(threshold_recall[0]),
            "error_f1": float(threshold_f1[0]),
            "balanced_accuracy": float(threshold_balanced_accuracy),
        }
        print(
            f"threshold={threshold:.2f} | error_precision={current['error_precision']:.4f} | "
            f"error_recall={current['error_recall']:.4f} | error_F1={current['error_f1']:.4f} | "
            f"balanced_accuracy={current['balanced_accuracy']:.4f}"
        )
        if best_result is None or current["error_f1"] > best_result["error_f1"]:
            best_threshold = threshold
            best_result = current

    print(
        f"Best threshold by error F1: {best_threshold:.2f} | "
        f"error_precision={best_result['error_precision']:.4f} | "
        f"error_recall={best_result['error_recall']:.4f} | "
        f"error_F1={best_result['error_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
