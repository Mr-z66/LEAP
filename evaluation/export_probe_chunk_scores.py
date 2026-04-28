import argparse
import csv
import os
import re

import numpy as np
import torch


class TorchMLPProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.0):
        super().__init__()
        dims = [input_dim, *hidden_layers, 1]
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
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
    parser = argparse.ArgumentParser(description="Export held-out chunk-level probe scores to CSV.")
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk dataset (.pt).")
    parser.add_argument("--probe-artifact-path", required=True, help="Path to trained probe artifact (.pt).")
    parser.add_argument("--output-path", required=True, help="CSV output path.")
    parser.add_argument("--feature-set-name", required=True, help="Short name written into the CSV, e.g. hidden_only.")
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
    for part in str(feature_spec).split("+"):
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
        elif token in DERIVED_SCALAR_FEATURES:
            value = chunk_scalar_feature(chunk, token)
        else:
            feature_key = canonical_feature_name(token)
            if feature_key not in chunk:
                raise KeyError(f"Feature component '{token}' (resolved to '{feature_key}') not found in chunk.")
            value = tensor_to_numpy(chunk[feature_key])
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(values, axis=0)


def build_question_records(dataset, feature_spec, label_key):
    required_tokens = parse_feature_spec(feature_spec)
    question_records = {}

    if dataset and isinstance(dataset[0], dict) and "chunks" in dataset[0]:
        for item in dataset:
            question_id = int(item["question_id"])
            chunks = []
            has_all_components = True
            for chunk in item.get("chunks", []):
                if label_key not in chunk:
                    has_all_components = False
                    break
                for token in required_tokens:
                    if token in {"delta_prev", "abs_delta_prev"} or token in DERIVED_SCALAR_FEATURES:
                        continue
                    resolved_key = canonical_feature_name(token)
                    if resolved_key not in chunk:
                        has_all_components = False
                        break
                if not has_all_components:
                    break
                chunks.append(chunk)
            if has_all_components and chunks:
                question_records[question_id] = {"chunks": sorted(chunks, key=lambda c: int(c["chunk_id"]))}
        return question_records

    for item in dataset:
        if label_key not in item:
            continue
        has_all_components = True
        for token in required_tokens:
            if token in {"delta_prev", "abs_delta_prev"} or token in DERIVED_SCALAR_FEATURES:
                continue
            resolved_key = canonical_feature_name(token)
            if resolved_key not in item:
                has_all_components = False
                break
        if not has_all_components:
            continue
        question_id = int(item["question_id"])
        record = question_records.setdefault(question_id, {"chunks": []})
        record["chunks"].append(item)

    for record in question_records.values():
        record["chunks"] = sorted(record["chunks"], key=lambda c: int(c["chunk_id"]))
    return question_records


def artifact_positive_prob_to_trigger_score(artifact, positive_prob):
    label_key = artifact.get("label_key", "label")
    if label_key == "takeover_beneficial":
        return positive_prob
    return 1.0 - positive_prob


def parse_hidden_layer_sizes(hidden_layers_text):
    layer_sizes = []
    for part in str(hidden_layers_text).split(","):
        part = part.strip()
        if not part:
            continue
        layer_sizes.append(int(part))
    if not layer_sizes:
        raise ValueError("Hidden layers cannot be empty when reconstructing a TorchMLPProbe.")
    return tuple(layer_sizes)


def load_probe_artifact(path):
    artifact = torch.load(path, weights_only=False)
    probe = artifact.get("probe")
    if probe is None and artifact.get("probe_state_dict") is not None:
        hidden_layers = parse_hidden_layer_sizes(artifact["config"]["hidden_layers"])
        dropout = float(artifact.get("config", {}).get("dropout", 0.0))
        probe = TorchMLPProbe(
            input_dim=int(artifact["feature_dim"]),
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        probe.load_state_dict(artifact["probe_state_dict"])
        artifact["probe"] = probe
    return artifact


def main():
    args = parse_args()

    artifact = load_probe_artifact(args.probe_artifact_path)
    probe = artifact["probe"]
    scaler = artifact["scaler"]
    feature_spec = artifact["feature_key"]
    label_key = artifact.get("label_key", "label")
    test_question_ids = {int(qid) for qid in artifact["test_question_ids"]}

    dataset = torch.load(args.label_path, weights_only=False)
    question_records = build_question_records(dataset, feature_spec, label_key)

    rows = []
    for question_id in sorted(test_question_ids):
        record = question_records.get(question_id)
        if record is None:
            continue
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            feature_vector = build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec)
            features = scaler.transform([feature_vector])
            positive_prob = float(probe.predict_proba(features)[0, 1])
            trigger_score = float(artifact_positive_prob_to_trigger_score(artifact, positive_prob))
            rows.append(
                {
                    "feature_set": args.feature_set_name,
                    "question_id": int(question_id),
                    "chunk_id": int(chunk["chunk_id"]),
                    "label": int(chunk[label_key]),
                    "positive_prob": positive_prob,
                    "trigger_score": trigger_score,
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature_set", "question_id", "chunk_id", "label", "positive_prob", "trigger_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} chunk rows to: {args.output_path}")
    print(f"Feature set: {args.feature_set_name}")
    print(f"Artifact feature spec: {feature_spec}")
    print(f"Label key: {label_key}")


if __name__ == "__main__":
    main()
