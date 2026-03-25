import os
import statistics

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data.pt")
FEATURE_KEYS = [
    "mean_hidden_state",
    "boundary_hidden_state",
]
TEST_SIZE = 0.2
NUM_SPLITS = 20
BASE_RANDOM_STATE = 42
# =================================================


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


def run_single_split(X, y, groups, random_state):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
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

    return {
        "random_state": random_state,
        "train_questions": len(np.unique(groups_train)),
        "test_questions": len(np.unique(groups_test)),
        "train_chunks": len(train_indices),
        "test_chunks": len(test_indices),
        "train_positive_ratio": float(y_train.mean()),
        "test_positive_ratio": float(y_test.mean()),
        "auroc": roc_auc_score(y_test, y_score),
        "auprc": average_precision_score(y_test, y_score),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }


def evaluate_feature(samples, feature_key):
    X = np.stack([tensor_to_numpy(sample[feature_key]) for sample in samples])
    y = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    groups = np.array([int(sample["question_id"]) for sample in samples], dtype=np.int64)

    split_results = []
    for offset in range(NUM_SPLITS):
        random_state = BASE_RANDOM_STATE + offset
        split_result = run_single_split(X, y, groups, random_state)
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


print(f"Loading labeled chunk dataset from: {DATA_PATH}")
dataset = torch.load(DATA_PATH)

all_results = []
for feature_key in FEATURE_KEYS:
    samples = load_filtered_samples(dataset, feature_key)
    if not samples:
        print(f"\nSkipping {feature_key}: no labeled samples found.")
        continue

    result = evaluate_feature(samples, feature_key)
    all_results.append(result)

    print("\nLeakage-free heuristic baseline")
    print("=" * 50)
    print(f"Feature: {result['feature_key']}")
    print(
        f"Chunks: {result['num_chunks']} | Questions: {result['num_questions']} "
        f"| Positive ratio: {result['positive_ratio']:.4f} | Negative ratio: {result['negative_ratio']:.4f}"
    )
    print(f"Valid grouped splits: {len(result['valid_splits'])}/{NUM_SPLITS}")

    if not result["valid_splits"]:
        print("No valid grouped split contained both classes in train and test.")
        print("Recommendation: increase labeled question count or negative examples before probe evaluation.")
        continue

    aurocs = [item["auroc"] for item in result["valid_splits"]]
    auprcs = [item["auprc"] for item in result["valid_splits"]]
    test_positive_ratios = [item["test_positive_ratio"] for item in result["valid_splits"]]

    print(f"AUROC mean/std: {safe_mean(aurocs):.4f} / {safe_stdev(aurocs):.4f}")
    print(f"AUPRC mean/std: {safe_mean(auprcs):.4f} / {safe_stdev(auprcs):.4f}")
    print(
        f"Test positive ratio mean/std: {safe_mean(test_positive_ratios):.4f} / "
        f"{safe_stdev(test_positive_ratios):.4f}"
    )

    best_split = max(result["valid_splits"], key=lambda item: item["auroc"])
    print("\nBest valid split snapshot")
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
    print(f"Best-split AUROC: {best_split['auroc']:.4f}")
    print(f"Best-split AUPRC: {best_split['auprc']:.4f}")
    print(best_split["classification_report"])

print("\nFeature comparison summary")
print("=" * 50)
for result in all_results:
    if not result["valid_splits"]:
        print(f"{result['feature_key']}: no valid grouped split")
        continue

    aurocs = [item["auroc"] for item in result["valid_splits"]]
    auprcs = [item["auprc"] for item in result["valid_splits"]]
    print(
        f"{result['feature_key']}: "
        f"AUROC={safe_mean(aurocs):.4f}+/-{safe_stdev(aurocs):.4f}, "
        f"AUPRC={safe_mean(auprcs):.4f}+/-{safe_stdev(auprcs):.4f}, "
        f"valid_splits={len(result['valid_splits'])}/{NUM_SPLITS}"
    )
