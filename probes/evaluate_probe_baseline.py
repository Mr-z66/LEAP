import argparse
import csv
import os
import re
import statistics

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_FEATURE_KEYS = [
    "boundary",
]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_NUM_SPLITS = 20
DEFAULT_BASE_RANDOM_STATE = 42
DEFAULT_PROBE_TYPE = "logistic"
DEFAULT_MLP_HIDDEN_LAYERS = "512,128,32"
DEFAULT_MLP_MAX_ITER = 300
DEFAULT_MLP_ALPHA = 1e-4
DEFAULT_MLP_LEARNING_RATE_INIT = 1e-3
DEFAULT_THRESHOLD_GRID = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
DEFAULT_SUMMARY_CSV_PATH = os.path.join(PROJECT_ROOT, "probe_feature_summary.csv")
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate probe baselines on labeled chunk datasets.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the labeled chunk dataset.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURE_KEYS,
        help=(
            "Feature specs to evaluate. Examples: boundary, mean, boundary+mean, "
            "boundary+delta_prev, boundary+mean+relative_position"
        ),
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="GroupShuffleSplit test ratio.")
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS, help="Number of grouped random splits.")
    parser.add_argument("--base-random-state", type=int, default=DEFAULT_BASE_RANDOM_STATE, help="Base random seed.")
    parser.add_argument(
        "--probe-type",
        choices=["logistic", "mlp"],
        default=DEFAULT_PROBE_TYPE,
        help="Probe model to train. Use logistic for the linear baseline or mlp for a multi-layer probe.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        default=DEFAULT_MLP_HIDDEN_LAYERS,
        help="Comma-separated hidden layer sizes for MLP, for example 512,128,32.",
    )
    parser.add_argument("--mlp-max-iter", type=int, default=DEFAULT_MLP_MAX_ITER, help="Max training iterations for MLP.")
    parser.add_argument("--mlp-alpha", type=float, default=DEFAULT_MLP_ALPHA, help="L2 regularization strength for MLP.")
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=DEFAULT_MLP_LEARNING_RATE_INIT,
        help="Initial learning rate for MLP.",
    )
    parser.add_argument(
        "--threshold-grid",
        default=DEFAULT_THRESHOLD_GRID,
        help="Comma-separated thresholds for scanning error-score decision points.",
    )
    parser.add_argument(
        "--summary-csv-path",
        default=DEFAULT_SUMMARY_CSV_PATH,
        help="Where to save the per-feature comparison summary CSV.",
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
        help="Repeat low-entropy strict error chunks this many times during MLP training. Use 1 to disable.",
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


def low_entropy_error_multiplier(row, args):
    if int(row["label"]) != 0:
        return 1
    if args.low_entropy_error_oversample <= 1:
        return 1

    entropy_max = args.low_entropy_error_final_entropy_max
    top1_min = args.low_entropy_error_final_top1_min
    if entropy_max is None and top1_min is None:
        return 1

    chunk = row["chunk"]
    entropy_value = float(tensor_to_numpy(chunk.get("final_entropy", 0.0)).reshape(-1)[0]) if "final_entropy" in chunk else None
    top1_value = float(tensor_to_numpy(chunk.get("final_top1_prob", 0.0)).reshape(-1)[0]) if "final_top1_prob" in chunk else None

    if entropy_max is not None and (entropy_value is None or entropy_value > entropy_max):
        return 1
    if top1_min is not None and (top1_value is None or top1_value < top1_min):
        return 1
    return args.low_entropy_error_oversample

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
        elif token in DERIVED_SCALAR_FEATURES:
            value = chunk_scalar_feature(chunk, token)
        else:
            feature_key = canonical_feature_name(token)
            if feature_key not in chunk:
                raise KeyError(f"Feature component '{token}' (resolved to '{feature_key}') not found in chunk.")
            value = tensor_to_numpy(chunk[feature_key])
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(values, axis=0)


def build_labeled_rows(question_records, feature_spec):
    rows = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(
                {
                    "question_id": question_id,
                    "label": int(chunk["label"]),
                    "features": build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec),
                    "chunk": chunk,
                }
            )
    return rows


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def safe_stdev(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def export_summary_csv(all_results, output_path):
    rows = []
    for result in all_results:
        if not result["valid_splits"]:
            rows.append(
                {
                    "feature_key": result["feature_key"],
                    "feature_dim": result["feature_dim"],
                    "valid_splits": 0,
                    "prefix_correct_auroc_mean": float("nan"),
                    "error_auprc_mean": float("nan"),
                    "error_f1_mean": float("nan"),
                    "best_threshold_error_f1_mean": float("nan"),
                }
            )
            continue

        prefix_correct_aurocs = [item["prefix_correct_auroc"] for item in result["valid_splits"]]
        error_auprcs = [item["error_auprc"] for item in result["valid_splits"]]
        error_f1s = [item["error_f1"] for item in result["valid_splits"]]
        best_threshold_f1s = [item["best_threshold_result"]["error_f1"] for item in result["valid_splits"]]
        rows.append(
            {
                "feature_key": result["feature_key"],
                "feature_dim": result["feature_dim"],
                "valid_splits": len(result["valid_splits"]),
                "prefix_correct_auroc_mean": safe_mean(prefix_correct_aurocs),
                "error_auprc_mean": safe_mean(error_auprcs),
                "error_f1_mean": safe_mean(error_f1s),
                "best_threshold_error_f1_mean": safe_mean(best_threshold_f1s),
            }
        )

    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_key",
                "feature_dim",
                "valid_splits",
                "prefix_correct_auroc_mean",
                "error_auprc_mean",
                "error_f1_mean",
                "best_threshold_error_f1_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_threshold_grid(threshold_text):
    thresholds = []
    for part in threshold_text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 < value < 1.0:
            raise ValueError("Thresholds must be between 0 and 1.")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("Threshold grid cannot be empty.")
    return sorted(set(thresholds))


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


def expand_by_multipliers(X, y, multipliers):
    if multipliers is None or np.all(multipliers == 1):
        return X, y
    repeated_indices = np.repeat(np.arange(len(y)), multipliers.astype(np.int64))
    return X[repeated_indices], y[repeated_indices]


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


def build_probe(args, random_state):
    if args.probe_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
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
        random_state=random_state,
    )


def compute_error_metrics_from_predictions(y_test, y_pred):
    y_true_error = 1 - y_test
    y_pred_error = 1 - y_pred
    error_precision, error_recall, error_f1, _ = precision_recall_fscore_support(
        y_true_error,
        y_pred_error,
        average="binary",
        zero_division=0,
    )
    return {
        "error_precision": error_precision,
        "error_recall": error_recall,
        "error_f1": error_f1,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
    }


def scan_thresholds(y_test, error_score, thresholds):
    threshold_results = []
    for threshold in thresholds:
        y_pred_error = (error_score >= threshold).astype(np.int64)
        y_pred = 1 - y_pred_error
        metrics = compute_error_metrics_from_predictions(y_test, y_pred)
        threshold_results.append(
            {
                "threshold": threshold,
                "error_precision": metrics["error_precision"],
                "error_recall": metrics["error_recall"],
                "error_f1": metrics["error_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    return threshold_results


def run_single_split(X, y, groups, multipliers, random_state, test_size, args):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups))

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    train_multipliers = multipliers[train_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = build_probe(args, random_state)
    if args.probe_type == "mlp":
        X_train_weighted, y_train_weighted = expand_by_multipliers(X_train_scaled, y_train, train_multipliers)
        X_train_fit, y_train_fit = upsample_minority_class(X_train_weighted, y_train_weighted, random_state)
        probe.fit(X_train_fit, y_train_fit)
    else:
        probe.fit(X_train_scaled, y_train)

    y_score = probe.predict_proba(X_test_scaled)[:, 1]
    y_pred = probe.predict(X_test_scaled)
    error_score = 1.0 - y_score
    y_true_error = 1 - y_test
    default_metrics = compute_error_metrics_from_predictions(y_test, y_pred)
    threshold_results = scan_thresholds(y_test, error_score, args.thresholds)
    best_threshold_result = max(threshold_results, key=lambda item: item["error_f1"])

    return {
        "random_state": random_state,
        "low_entropy_hard_negative_train_count": int(np.sum(train_multipliers > 1)),
        "train_questions": len(np.unique(groups_train)),
        "test_questions": len(np.unique(groups_test)),
        "train_chunks": len(train_indices),
        "test_chunks": len(test_indices),
        "train_positive_ratio": float(y_train.mean()),
        "test_positive_ratio": float(y_test.mean()),
        "probe_type": args.probe_type,
        "prefix_correct_auroc": roc_auc_score(y_test, y_score),
        "prefix_correct_auprc": average_precision_score(y_test, y_score),
        "error_auprc": average_precision_score(y_true_error, error_score),
        "balanced_accuracy": default_metrics["balanced_accuracy"],
        "error_precision": default_metrics["error_precision"],
        "error_recall": default_metrics["error_recall"],
        "error_f1": default_metrics["error_f1"],
        "confusion_matrix": default_metrics["confusion_matrix"],
        "classification_report": default_metrics["classification_report"],
        "threshold_results": threshold_results,
        "best_threshold_result": best_threshold_result,
    }


def evaluate_feature(samples, feature_spec, num_splits, base_random_state, test_size, args):
    X = np.stack([sample["features"] for sample in samples])
    y = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    groups = np.array([int(sample["question_id"]) for sample in samples], dtype=np.int64)
    multipliers = np.array([low_entropy_error_multiplier(sample, args) for sample in samples], dtype=np.int64)

    split_results = []
    for offset in range(num_splits):
        random_state = base_random_state + offset
        split_result = run_single_split(X, y, groups, multipliers, random_state, test_size, args)
        if split_result is not None:
            split_results.append(split_result)

    return {
        "feature_key": feature_spec,
        "feature_dim": int(X.shape[1]),
        "num_chunks": len(samples),
        "num_questions": len(np.unique(groups)),
        "positive_ratio": float(y.mean()),
        "negative_ratio": float(1.0 - y.mean()),
        "low_entropy_hard_negative_count": int(np.sum(multipliers > 1)),
        "valid_splits": split_results,
    }


args = parse_args()
args.thresholds = parse_threshold_grid(args.threshold_grid)

print(f"Loading labeled chunk dataset from: {args.data_path}")
dataset = torch.load(args.data_path)
question_records = build_question_records(dataset)

if args.probe_type == "mlp":
    print(
        f"Using MLP probe with hidden layers {parse_hidden_layer_sizes(args.mlp_hidden_layers)}, "
        f"max_iter={args.mlp_max_iter}, alpha={args.mlp_alpha}, "
        f"learning_rate_init={args.mlp_learning_rate_init}"
    )
else:
    print("Using logistic probe with class-balanced training.")
print(f"Threshold scan grid: {args.thresholds}")

all_results = []
for feature_key in args.features:
    samples = build_labeled_rows(question_records, feature_key)
    if not samples:
        print(f"\nSkipping {feature_key}: no labeled samples found.")
        continue

    result = evaluate_feature(
        samples=samples,
        feature_spec=feature_key,
        num_splits=args.num_splits,
        base_random_state=args.base_random_state,
        test_size=args.test_size,
        args=args,
    )
    all_results.append(result)

    print("\nLeakage-free strict baseline")
    print("=" * 50)
    print(f"Feature: {result['feature_key']} | Probe: {args.probe_type} | Dim: {result['feature_dim']}")
    print(
        f"Chunks: {result['num_chunks']} | Questions: {result['num_questions']} "
        f"| Positive ratio: {result['positive_ratio']:.4f} | Negative ratio: {result['negative_ratio']:.4f}"
    )
    if args.low_entropy_error_oversample > 1:
        print(
            f"Low-entropy hard negatives in dataset: {result['low_entropy_hard_negative_count']} | "
            f"oversample factor: {args.low_entropy_error_oversample}"
        )
    print(f"Valid grouped splits: {len(result['valid_splits'])}/{args.num_splits}")

    if not result["valid_splits"]:
        print("No valid grouped split contained both classes in train and test.")
        print("Recommendation: increase labeled question count or negative examples before probe evaluation.")
        continue

    prefix_correct_aurocs = [item["prefix_correct_auroc"] for item in result["valid_splits"]]
    prefix_correct_auprcs = [item["prefix_correct_auprc"] for item in result["valid_splits"]]
    error_auprcs = [item["error_auprc"] for item in result["valid_splits"]]
    balanced_accuracies = [item["balanced_accuracy"] for item in result["valid_splits"]]
    error_precisions = [item["error_precision"] for item in result["valid_splits"]]
    error_recalls = [item["error_recall"] for item in result["valid_splits"]]
    error_f1s = [item["error_f1"] for item in result["valid_splits"]]
    best_threshold_precisions = [item["best_threshold_result"]["error_precision"] for item in result["valid_splits"]]
    best_threshold_recalls = [item["best_threshold_result"]["error_recall"] for item in result["valid_splits"]]
    best_threshold_f1s = [item["best_threshold_result"]["error_f1"] for item in result["valid_splits"]]

    print(f"Prefix-correct AUROC mean/std: {safe_mean(prefix_correct_aurocs):.4f} / {safe_stdev(prefix_correct_aurocs):.4f}")
    print(f"Prefix-correct AUPRC mean/std: {safe_mean(prefix_correct_auprcs):.4f} / {safe_stdev(prefix_correct_auprcs):.4f}")
    print(f"Error-class AUPRC mean/std: {safe_mean(error_auprcs):.4f} / {safe_stdev(error_auprcs):.4f}")
    print(f"Balanced accuracy mean/std: {safe_mean(balanced_accuracies):.4f} / {safe_stdev(balanced_accuracies):.4f}")
    print(f"Error precision mean/std: {safe_mean(error_precisions):.4f} / {safe_stdev(error_precisions):.4f}")
    print(f"Error recall mean/std: {safe_mean(error_recalls):.4f} / {safe_stdev(error_recalls):.4f}")
    print(f"Error F1 mean/std: {safe_mean(error_f1s):.4f} / {safe_stdev(error_f1s):.4f}")
    print(f"Best-threshold error precision mean/std: {safe_mean(best_threshold_precisions):.4f} / {safe_stdev(best_threshold_precisions):.4f}")
    print(f"Best-threshold error recall mean/std: {safe_mean(best_threshold_recalls):.4f} / {safe_stdev(best_threshold_recalls):.4f}")
    print(f"Best-threshold error F1 mean/std: {safe_mean(best_threshold_f1s):.4f} / {safe_stdev(best_threshold_f1s):.4f}")

    best_split = max(result["valid_splits"], key=lambda item: item["error_f1"])
    print("\nBest valid split snapshot (by error F1)")
    print("-" * 50)
    print(
        f"Seed: {best_split['random_state']} | Train questions: {best_split['train_questions']} "
        f"| Test questions: {best_split['test_questions']}"
    )
    print(
        f"Train chunks: {best_split['train_chunks']} | Test chunks: {best_split['test_chunks']} "
        f"| Train positive ratio: {best_split['train_positive_ratio']:.4f} "
        f"| Test positive ratio: {best_split['test_positive_ratio']:.4f}"
    )
    print(f"Prefix-correct AUROC: {best_split['prefix_correct_auroc']:.4f}")
    print(f"Prefix-correct AUPRC: {best_split['prefix_correct_auprc']:.4f}")
    print(f"Error-class AUPRC: {best_split['error_auprc']:.4f}")
    print(f"Balanced accuracy: {best_split['balanced_accuracy']:.4f}")
    print(f"Error precision: {best_split['error_precision']:.4f}")
    print(f"Error recall: {best_split['error_recall']:.4f}")
    print(f"Error F1: {best_split['error_f1']:.4f}")
    print(f"Confusion matrix [true 0/1 x pred 0/1]: {best_split['confusion_matrix']}")
    print(best_split["classification_report"])
    print("\nThreshold scan on best split")
    print("-" * 50)
    for item in best_split["threshold_results"]:
        print(
            f"threshold={item['threshold']:.2f} | "
            f"error_precision={item['error_precision']:.4f} | "
            f"error_recall={item['error_recall']:.4f} | "
            f"error_F1={item['error_f1']:.4f} | "
            f"balanced_accuracy={item['balanced_accuracy']:.4f}"
        )
    print(
        "Best threshold by error F1: "
        f"{best_split['best_threshold_result']['threshold']:.2f} | "
        f"error_precision={best_split['best_threshold_result']['error_precision']:.4f} | "
        f"error_recall={best_split['best_threshold_result']['error_recall']:.4f} | "
        f"error_F1={best_split['best_threshold_result']['error_f1']:.4f}"
    )

print("\nFeature comparison summary")
print("=" * 50)
for result in all_results:
    if not result["valid_splits"]:
        print(f"{result['feature_key']}: no valid grouped split")
        continue

    prefix_correct_aurocs = [item["prefix_correct_auroc"] for item in result["valid_splits"]]
    error_auprcs = [item["error_auprc"] for item in result["valid_splits"]]
    error_recalls = [item["error_recall"] for item in result["valid_splits"]]
    error_f1s = [item["error_f1"] for item in result["valid_splits"]]
    print(
        f"{result['feature_key']}: "
        f"dim={result['feature_dim']}, "
        f"prefix_AUROC={safe_mean(prefix_correct_aurocs):.4f}+/-{safe_stdev(prefix_correct_aurocs):.4f}, "
        f"error_AUPRC={safe_mean(error_auprcs):.4f}+/-{safe_stdev(error_auprcs):.4f}, "
        f"error_recall={safe_mean(error_recalls):.4f}+/-{safe_stdev(error_recalls):.4f}, "
        f"error_F1={safe_mean(error_f1s):.4f}+/-{safe_stdev(error_f1s):.4f}, "
        f"valid_splits={len(result['valid_splits'])}/{args.num_splits}"
    )

if all_results:
    ranked_results = [result for result in all_results if result["valid_splits"]]
    if ranked_results:
        best_feature = max(
            ranked_results,
            key=lambda item: safe_mean([split["error_f1"] for split in item["valid_splits"]]),
        )
        best_error_f1 = safe_mean([split["error_f1"] for split in best_feature["valid_splits"]])
        print("\nRecommended primary feature")
        print("=" * 50)
        print(
            f"{best_feature['feature_key']} | dim={best_feature['feature_dim']} | "
            f"probe={args.probe_type} | mean error_F1={best_error_f1:.4f} | "
            f"dataset={os.path.basename(args.data_path)}"
        )

export_summary_csv(all_results, args.summary_csv_path)
print(f"\nSaved feature comparison summary to: {args.summary_csv_path}")

