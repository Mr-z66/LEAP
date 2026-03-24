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
FEATURE_KEY = "mean_hidden_state"  # or "boundary_hidden_state"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =================================================


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


print(f"Loading labeled chunk dataset from: {DATA_PATH}")
dataset = torch.load(DATA_PATH)

filtered_samples = [sample for sample in dataset if FEATURE_KEY in sample and sample["label"] in {0, 1}]
if not filtered_samples:
    raise ValueError("No labeled samples found. Run referee_32b_labeling.py first.")

X = np.stack([tensor_to_numpy(sample[FEATURE_KEY]) for sample in filtered_samples])
y = np.array([int(sample["label"]) for sample in filtered_samples], dtype=np.int64)
groups = np.array([int(sample["question_id"]) for sample in filtered_samples], dtype=np.int64)

unique_questions = np.unique(groups)
print(f"Chunks: {len(filtered_samples)} | Questions: {len(unique_questions)} | Positive ratio: {y.mean():.4f}")

group_splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_indices, test_indices = next(group_splitter.split(X, y, groups))

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
groups_train = groups[train_indices]
groups_test = groups[test_indices]

print(
    f"Train questions: {len(np.unique(groups_train))} | Test questions: {len(np.unique(groups_test))} "
    f"| Train chunks: {len(train_indices)} | Test chunks: {len(test_indices)}"
)

if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
    raise ValueError("Grouped split produced a single-class train/test set. Increase data size or adjust test split.")

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

auroc = roc_auc_score(y_test, y_score)
auprc = average_precision_score(y_test, y_score)

print("\nLeakage-free heuristic baseline")
print("=" * 50)
print(f"Feature: {FEATURE_KEY}")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(classification_report(y_test, y_pred, digits=4))
