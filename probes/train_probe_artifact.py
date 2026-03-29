import argparse
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "probe_artifact.pt")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 55
DEFAULT_FEATURE_KEY = "boundary"
DEFAULT_PROBE_TYPE = "mlp"
DEFAULT_LABEL_KEY = "label"
DEFAULT_MLP_HIDDEN_LAYERS = "512,128,32"
DEFAULT_MLP_MAX_ITER = 300
DEFAULT_MLP_ALPHA = 1e-4
DEFAULT_MLP_LEARNING_RATE_INIT = 1e-3
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save a fixed probe artifact for routing experiments.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to labeled chunk data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Where to save the probe artifact.")
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Feature spec used by the probe, for example boundary+mean+relative_position.",
    )
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY, help="Label field to train on, for example label or takeover_beneficial.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Held-out question ratio.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for the split.")
    parser.add_argument(
        "--probe-type",
        choices=["logistic", "mlp"],
        default=DEFAULT_PROBE_TYPE,
        help="Probe type used for risk scoring.",
    )
    parser.add_argument("--mlp-hidden-layers", default=DEFAULT_MLP_HIDDEN_LAYERS, help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--mlp-max-iter", type=int, default=DEFAULT_MLP_MAX_ITER, help="MLP max iterations.")
    parser.add_argument("--mlp-alpha", type=float, default=DEFAULT_MLP_ALPHA, help="MLP L2 regularization.")
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=DEFAULT_MLP_LEARNING_RATE_INIT,
        help="MLP initial learning rate.",
    )
    parser.add_argument(
        "--low-entropy-error-final-entropy-max",
        type=float,
        default=None,
        help="Optional max final_entropy for treating strict error chunks as low-entropy hard negatives.",
    )
    parser.add_argument(
        "--low-entropy-error-final-top1-min",
        type=float,
        default=None,
        help="Optional min final_top1_prob for treating strict error chunks as low-entropy hard negatives.",
    )
    parser.add_argument(
        "--low-entropy-error-oversample",
        type=int,
        default=1,
        help="Repeat low-entropy strict error chunks this many times during training. Use 1 to disable.",
    )
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


def low_entropy_error_multiplier(chunk, label_key, args):
    if label_key != "label":
        return 1
    if int(chunk.get(label_key, 1)) != 0:
        return 1
    if args.low_entropy_error_oversample <= 1:
        return 1

    entropy_max = args.low_entropy_error_final_entropy_max
    top1_min = args.low_entropy_error_final_top1_min
    if entropy_max is None and top1_min is None:
        return 1

    entropy_value = float(tensor_to_numpy(chunk.get("final_entropy", 0.0)).reshape(-1)[0]) if "final_entropy" in chunk else None
    top1_value = float(tensor_to_numpy(chunk.get("final_top1_prob", 0.0)).reshape(-1)[0]) if "final_top1_prob" in chunk else None

    if entropy_max is not None and (entropy_value is None or entropy_value > entropy_max):
        return 1
    if top1_min is not None and (top1_value is None or top1_value < top1_min):
        return 1
    return args.low_entropy_error_oversample

def build_question_records(dataset, label_key):
    question_records = {}
    for item in dataset:
        if label_key not in item:
            continue
        label_value = int(item[label_key])
        if label_value not in {0, 1}:
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
        elif token in DERIVED_SCALAR_FEATURES:
            value = chunk_scalar_feature(chunk, token)
        else:
            feature_key = canonical_feature_name(token)
            if feature_key not in chunk:
                raise KeyError(f"Feature component '{token}' (resolved to '{feature_key}') not found in chunk.")
            value = tensor_to_numpy(chunk[feature_key])
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(values, axis=0)


def build_feature_arrays(question_records, feature_spec, label_key, args=None):
    rows = []
    labels = []
    groups = []
    multipliers = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec))
            labels.append(int(chunk[label_key]))
            groups.append(question_id)
            multipliers.append(low_entropy_error_multiplier(chunk, label_key, args) if args is not None else 1)
    return np.stack(rows), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64), np.asarray(multipliers, dtype=np.int64)


def parse_hidden_layer_sizes(hidden_layers_text):
    layer_sizes = []
    for part in hidden_layers_text.split(","):
        part = part.strip()
        if not part:
            continue
        layer_sizes.append(int(part))
    if not layer_sizes:
        raise ValueError("MLP hidden layers cannot be empty.")
    return tuple(layer_sizes)


def expand_by_multipliers(X, y, multipliers, groups=None):
    if multipliers is None or np.all(multipliers == 1):
        if groups is None:
            return X, y
        return X, y, groups

    repeated_indices = np.repeat(np.arange(len(y)), multipliers.astype(np.int64))
    if groups is None:
        return X[repeated_indices], y[repeated_indices]
    return X[repeated_indices], y[repeated_indices], groups[repeated_indices]


def upsample_minority_class(X, y, random_state):
    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2 or counts[0] == counts[1]:
        return X, y

    majority_label = unique_labels[np.argmax(counts)]
    minority_label = unique_labels[np.argmin(counts)]
    majority_indices = np.flatnonzero(y == majority_label)
    minority_indices = np.flatnonzero(y == minority_label)

    rng = np.random.default_rng(random_state)
    sampled_minority_indices = rng.choice(minority_indices, size=len(majority_indices), replace=True)
    balanced_indices = np.concatenate([majority_indices, sampled_minority_indices])
    rng.shuffle(balanced_indices)
    return X[balanced_indices], y[balanced_indices]


def build_probe(args):
    if args.probe_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_state,
        )

    return MLPClassifier(
        hidden_layer_sizes=parse_hidden_layer_sizes(args.mlp_hidden_layers),
        activation="relu",
        solver="adam",
        alpha=args.mlp_alpha,
        batch_size="auto",
        learning_rate_init=args.mlp_learning_rate_init,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=args.random_state,
    )


def main():
    args = parse_args()

    print(f"Loading labeled chunk dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)
    question_records = build_question_records(dataset, args.label_key)
    print(f"Loaded questions with feature spec '{args.feature_key}' and label '{args.label_key}': {len(question_records)}")

    X, y, groups, multipliers = build_feature_arrays(question_records, args.feature_key, args.label_key, args=args)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    if X.shape[0] == 0:
        raise ValueError(f"No training rows found for label_key={args.label_key!r} and feature_key={args.feature_key!r}.")
    if len(unique_labels) < 2:
        raise ValueError(
            f"Label {args.label_key!r} has only one class in the dataset: "
            f"labels={unique_labels.tolist()} counts={label_counts.tolist()}"
        )
    if len(set(int(qid) for qid in groups)) < 2:
        raise ValueError("Need at least two question groups to make a grouped train/test split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_indices])
    y_train = y[train_indices]
    train_multipliers = multipliers[train_indices]

    probe = build_probe(args)
    if args.probe_type == "mlp":
        X_weighted, y_weighted = expand_by_multipliers(X_train, y_train, train_multipliers)
        X_fit, y_fit = upsample_minority_class(X_weighted, y_weighted, args.random_state)
        probe.fit(X_fit, y_fit)
    else:
        probe.fit(X_train, y_train)

    train_question_ids = sorted(set(int(qid) for qid in groups[train_indices]))
    test_question_ids = sorted(set(int(qid) for qid in groups[test_indices]))

    artifact = {
        "probe": probe,
        "scaler": scaler,
        "feature_key": args.feature_key,
        "feature_dim": int(X.shape[1]),
        "label_key": args.label_key,
        "probe_type": args.probe_type,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "config": {
            "mlp_hidden_layers": args.mlp_hidden_layers,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_alpha": args.mlp_alpha,
            "mlp_learning_rate_init": args.mlp_learning_rate_init,
        },
    }

    torch.save(artifact, args.output_path)
    if args.low_entropy_error_oversample > 1 and args.label_key == "label":
        low_entropy_train_count = int(np.sum(train_multipliers > 1))
        print(
            "Low-entropy hard negative oversampling | "
            f"count={low_entropy_train_count} | factor={args.low_entropy_error_oversample}"
        )
    print(f"Saved probe artifact to: {args.output_path}")
    print(f"Feature dim: {artifact['feature_dim']}")
    print(f"Train questions: {len(train_question_ids)} | Test questions: {len(test_question_ids)}")


if __name__ == "__main__":
    main()

