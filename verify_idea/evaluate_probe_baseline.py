import os

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
RANDOM_STATE = 42
# =================================================


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def load_filtered_samples(dataset, feature_key):
    return [sample for sample in dataset if feature_key in sample and sample["label"] in {0, 1}]


def evaluate_feature(samples, feature_key):
    X = np.stack([tensor_to_numpy(sample[feature_key]) for sample in samples])
    y = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    groups = np.array([int(sample["question_id"]) for sample in samples], dtype=np.int64)

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(group_splitter.split(X, y, groups))

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            f"{feature_key}: grouped split produced a single-class train/test set. "
            "Increase data size or adjust test split."
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    probe.fit(X_train_scaled, y_train)

    y_score = probe.predict_proba(X_test_scaled)[:, 1]
    y_pred = probe.predict(X_test_scaled)

    return {
        "feature_key": feature_key,
        "num_chunks": len(samples),
        "num_questions": len(np.unique(groups)),
        "positive_ratio": y.mean(),
        "train_questions": len(np.unique(groups_train)),
        "test_questions": len(np.unique(groups_test)),
        "train_chunks": len(train_indices),
        "test_chunks": len(test_indices),
        "auroc": roc_auc_score(y_test, y_score),
        "auprc": average_precision_score(y_test, y_score),
        "classification_report": classification_report(y_test, y_pred, digits=4),
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
        f"| Positive ratio: {result['positive_ratio']:.4f}"
    )
    print(
        f"Train questions: {result['train_questions']} | Test questions: {result['test_questions']} "
        f"| Train chunks: {result['train_chunks']} | Test chunks: {result['test_chunks']}"
    )
    print(f"AUROC: {result['auroc']:.4f}")
    print(f"AUPRC: {result['auprc']:.4f}")
    print(result["classification_report"])

if len(all_results) >= 2:
    print("Feature comparison summary")
    print("=" * 50)
    for result in sorted(all_results, key=lambda item: item["auroc"], reverse=True):
        print(
            f"{result['feature_key']}: AUROC={result['auroc']:.4f}, "
            f"AUPRC={result['auprc']:.4f}"
        )
