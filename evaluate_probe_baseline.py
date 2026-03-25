import argparse
import os
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
from sklearn.preprocessing import StandardScaler

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_FEATURE_KEYS = [
    "boundary_hidden_state",
]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_NUM_SPLITS = 20
DEFAULT_BASE_RANDOM_STATE = 42
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate probe baselines on labeled chunk datasets.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the labeled chunk dataset.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURE_KEYS,
        help="Feature keys to evaluate. Defaults to boundary first, then mean.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="GroupShuffleSplit test ratio.")
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS, help="Number of grouped random splits.")
    parser.add_argument("--base-random-state", type=int, default=DEFAULT_BASE_RANDOM_STATE, help="Base random seed.")
    return parser.parse_args()


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def load_filtered_samples(dataset, feature_key):
    return [sample for sample in dataset if feature_key in sample and sample["label"] in {0, 1}]


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def safe_stdev(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def run_single_split(X, y, groups, random_state, test_size):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups))

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    probe.fit(X_train_scaled, y_train)

    y_score = probe.predict_proba(X_test_scaled)[:, 1]
    y_pred = probe.predict(X_test_scaled)
    error_score = 1.0 - y_score
    y_true_error = 1 - y_test
    y_pred_error = 1 - y_pred

    error_precision, error_recall, error_f1, _ = precision_recall_fscore_support(
        y_true_error,
        y_pred_error,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "random_state": random_state,
        "train_questions": len(np.unique(groups_train)),
        "test_questions": len(np.unique(groups_test)),
        "train_chunks": len(train_indices),
        "test_chunks": len(test_indices),
        "train_positive_ratio": float(y_train.mean()),
        "test_positive_ratio": float(y_test.mean()),
        "prefix_correct_auroc": roc_auc_score(y_test, y_score),
        "prefix_correct_auprc": average_precision_score(y_test, y_score),
        "error_auprc": average_precision_score(y_true_error, error_score),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "error_precision": error_precision,
        "error_recall": error_recall,
        "error_f1": error_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
    }


def evaluate_feature(samples, feature_key, num_splits, base_random_state, test_size):
    X = np.stack([tensor_to_numpy(sample[feature_key]) for sample in samples])
    y = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    groups = np.array([int(sample["question_id"]) for sample in samples], dtype=np.int64)

    split_results = []
    for offset in range(num_splits):
        random_state = base_random_state + offset
        split_result = run_single_split(X, y, groups, random_state, test_size)
        if split_result is not None:
            split_results.append(split_result)

    return {
        "feature_key": feature_key,
        "num_chunks": len(samples),
        "num_questions": len(np.unique(groups)),
        "positive_ratio": float(y.mean()),
        "negative_ratio": float(1.0 - y.mean()),
        "valid_splits": split_results,
    }


args = parse_args()

print(f"Loading labeled chunk dataset from: {args.data_path}")
dataset = torch.load(args.data_path)

all_results = []
for feature_key in args.features:
    samples = load_filtered_samples(dataset, feature_key)
    if not samples:
        print(f"\nSkipping {feature_key}: no labeled samples found.")
        continue

    result = evaluate_feature(
        samples=samples,
        feature_key=feature_key,
        num_splits=args.num_splits,
        base_random_state=args.base_random_state,
        test_size=args.test_size,
    )
    all_results.append(result)

    print("\nLeakage-free strict baseline")
    print("=" * 50)
    print(f"Feature: {result['feature_key']}")
    print(
        f"Chunks: {result['num_chunks']} | Questions: {result['num_questions']} "
        f"| Positive ratio: {result['positive_ratio']:.4f} | Negative ratio: {result['negative_ratio']:.4f}"
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

    print(f"Prefix-correct AUROC mean/std: {safe_mean(prefix_correct_aurocs):.4f} / {safe_stdev(prefix_correct_aurocs):.4f}")
    print(f"Prefix-correct AUPRC mean/std: {safe_mean(prefix_correct_auprcs):.4f} / {safe_stdev(prefix_correct_auprcs):.4f}")
    print(f"Error-class AUPRC mean/std: {safe_mean(error_auprcs):.4f} / {safe_stdev(error_auprcs):.4f}")
    print(f"Balanced accuracy mean/std: {safe_mean(balanced_accuracies):.4f} / {safe_stdev(balanced_accuracies):.4f}")
    print(f"Error precision mean/std: {safe_mean(error_precisions):.4f} / {safe_stdev(error_precisions):.4f}")
    print(f"Error recall mean/std: {safe_mean(error_recalls):.4f} / {safe_stdev(error_recalls):.4f}")
    print(f"Error F1 mean/std: {safe_mean(error_f1s):.4f} / {safe_stdev(error_f1s):.4f}")

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
        f"prefix_AUROC={safe_mean(prefix_correct_aurocs):.4f}+/-{safe_stdev(prefix_correct_aurocs):.4f}, "
        f"error_AUPRC={safe_mean(error_auprcs):.4f}+/-{safe_stdev(error_auprcs):.4f}, "
        f"error_recall={safe_mean(error_recalls):.4f}+/-{safe_stdev(error_recalls):.4f}, "
        f"error_F1={safe_mean(error_f1s):.4f}+/-{safe_stdev(error_f1s):.4f}, "
        f"valid_splits={len(result['valid_splits'])}/{args.num_splits}"
    )

if all_results:
    ranked_results = [
        result
        for result in all_results
        if result["valid_splits"]
    ]
    if ranked_results:
        best_feature = max(
            ranked_results,
            key=lambda item: safe_mean([split["error_f1"] for split in item["valid_splits"]]),
        )
        best_error_f1 = safe_mean([split["error_f1"] for split in best_feature["valid_splits"]])
        print("\nRecommended primary feature")
        print("=" * 50)
        print(
            f"{best_feature['feature_key']} | "
            f"mean error_F1={best_error_f1:.4f} | "
            f"dataset={os.path.basename(args.data_path)}"
        )
