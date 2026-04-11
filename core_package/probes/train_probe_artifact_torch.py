import argparse
import copy
import os
import re

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from core_package.config import PROBE_TRAIN


DEFAULT_LABEL_PATH = PROBE_TRAIN.label_path
DEFAULT_OUTPUT_PATH = PROBE_TRAIN.output_path
DEFAULT_TEST_SIZE = PROBE_TRAIN.test_size
DEFAULT_VAL_SIZE = PROBE_TRAIN.val_size
DEFAULT_RANDOM_STATE = PROBE_TRAIN.random_state
DEFAULT_FEATURE_KEY = PROBE_TRAIN.feature_key
DEFAULT_LABEL_KEY = PROBE_TRAIN.label_key
DEFAULT_HIDDEN_LAYERS = PROBE_TRAIN.hidden_layers
DEFAULT_DROPOUT = PROBE_TRAIN.dropout
DEFAULT_EPOCHS = PROBE_TRAIN.epochs
DEFAULT_BATCH_SIZE = PROBE_TRAIN.batch_size
DEFAULT_LEARNING_RATE = PROBE_TRAIN.learning_rate
DEFAULT_WEIGHT_DECAY = PROBE_TRAIN.weight_decay
DEFAULT_POS_WEIGHT = PROBE_TRAIN.pos_weight
DEFAULT_LOW_ENTROPY_ERROR_WEIGHT = PROBE_TRAIN.low_entropy_error_weight
DEFAULT_EARLY_STOPPING_PATIENCE = PROBE_TRAIN.early_stopping_patience
DEFAULT_MIN_EPOCHS = PROBE_TRAIN.min_epochs


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save a PyTorch probe artifact for routing experiments.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to labeled chunk data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Where to save the probe artifact.")
    parser.add_argument("--feature-key", default=DEFAULT_FEATURE_KEY, help="Feature spec used by the probe.")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY, help="Label field to train on.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Held-out question ratio.")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE, help="Validation ratio within training questions.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for the split.")
    parser.add_argument("--hidden-layers", default=DEFAULT_HIDDEN_LAYERS, help="Comma-separated hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability between hidden layers.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--pos-weight", type=float, default=DEFAULT_POS_WEIGHT, help="Positive-class BCE weight.")
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="Epochs to wait for validation improvement before stopping.")
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS, help="Minimum epochs before early stopping can trigger.")
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
        "--low-entropy-error-weight",
        type=float,
        default=DEFAULT_LOW_ENTROPY_ERROR_WEIGHT,
        help="Sample weight multiplier for strict low-entropy error chunks.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device. Use cpu by default because features are small.",
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


def low_entropy_error_weight(chunk, label_key, args):
    if label_key != "label":
        return 1.0
    if int(chunk.get(label_key, 1)) != 0:
        return 1.0

    entropy_max = args.low_entropy_error_final_entropy_max
    top1_min = args.low_entropy_error_final_top1_min
    if entropy_max is None and top1_min is None:
        return 1.0

    entropy_value = float(tensor_to_numpy(chunk.get("final_entropy", 0.0)).reshape(-1)[0]) if "final_entropy" in chunk else None
    top1_value = float(tensor_to_numpy(chunk.get("final_top1_prob", 0.0)).reshape(-1)[0]) if "final_top1_prob" in chunk else None

    if entropy_max is not None and (entropy_value is None or entropy_value > entropy_max):
        return 1.0
    if top1_min is not None and (top1_value is None or top1_value < top1_min):
        return 1.0
    return float(args.low_entropy_error_weight)


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


def build_feature_arrays(question_records, feature_spec, label_key, args):
    rows = []
    labels = []
    groups = []
    sample_weights = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec))
            labels.append(int(chunk[label_key]))
            groups.append(question_id)
            sample_weights.append(low_entropy_error_weight(chunk, label_key, args))
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
        np.asarray(sample_weights, dtype=np.float32),
    )


def parse_hidden_layer_sizes(hidden_layers_text):
    layer_sizes = []
    for part in hidden_layers_text.split(","):
        part = part.strip()
        if not part:
            continue
        layer_sizes.append(int(part))
    if not layer_sizes:
        raise ValueError("Hidden layers cannot be empty.")
    return tuple(layer_sizes)


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


def batch_indices(num_rows, batch_size, rng):
    indices = np.arange(num_rows)
    rng.shuffle(indices)
    for start in range(0, num_rows, batch_size):
        yield indices[start:start + batch_size]


def weighted_average_loss(model, criterion, X_tensor, y_tensor, weight_tensor, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, X_tensor.shape[0], batch_size):
            end = start + batch_size
            batch_x = X_tensor[start:end].to(device)
            batch_y = y_tensor[start:end].to(device)
            batch_w = weight_tensor[start:end].to(device)
            logits = model(batch_x)
            loss_per_sample = criterion(logits, batch_y)
            total_loss += float((loss_per_sample * batch_w).sum().item())
            total_weight += float(batch_w.sum().item())
    return total_loss / max(total_weight, 1e-8)


def main():
    args = parse_args()

    print(f"Loading labeled chunk dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)
    question_records = build_question_records(dataset, args.label_key)
    print(f"Loaded questions with feature spec '{args.feature_key}' and label '{args.label_key}': {len(question_records)}")

    X, y, groups, sample_weights = build_feature_arrays(question_records, args.feature_key, args.label_key, args=args)
    unique_labels, label_counts = np.unique(y.astype(np.int64), return_counts=True)
    if X.shape[0] == 0:
        raise ValueError(f"No training rows found for label_key={args.label_key!r} and feature_key={args.feature_key!r}.")
    if len(unique_labels) < 2:
        raise ValueError(
            f"Label {args.label_key!r} has only one class in the dataset: "
            f"labels={unique_labels.tolist()} counts={label_counts.tolist()}"
        )
    if len(set(int(qid) for qid in groups)) < 3:
        raise ValueError("Need at least three question groups to make train/val/test grouped splits.")

    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_pool_indices, test_indices = next(outer_splitter.split(X, y, groups))

    inner_groups = groups[train_pool_indices]
    if len(set(int(qid) for qid in inner_groups)) < 2:
        raise ValueError("Need at least two training question groups to make a validation split.")
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.random_state + 1)
    inner_train_rel, val_rel = next(inner_splitter.split(X[train_pool_indices], y[train_pool_indices], inner_groups))
    inner_train_indices = train_pool_indices[inner_train_rel]
    val_indices = train_pool_indices[val_rel]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[inner_train_indices]).astype(np.float32)
    X_val = scaler.transform(X[val_indices]).astype(np.float32)
    y_train = y[inner_train_indices].astype(np.float32)
    y_val = y[val_indices].astype(np.float32)
    train_sample_weights = sample_weights[inner_train_indices].astype(np.float32)
    val_sample_weights = sample_weights[val_indices].astype(np.float32)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    hidden_layers = parse_hidden_layer_sizes(args.hidden_layers)
    probe = TorchMLPProbe(input_dim=X.shape[1], hidden_layers=hidden_layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    rng = np.random.default_rng(args.random_state)

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    weight_train_tensor = torch.from_numpy(train_sample_weights)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)
    weight_val_tensor = torch.from_numpy(val_sample_weights)

    best_state_dict = copy.deepcopy(probe.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        probe.train()
        epoch_loss = 0.0
        epoch_weight = 0.0
        for batch_idx in batch_indices(X_train.shape[0], args.batch_size, rng):
            batch_x = X_train_tensor[batch_idx].to(device)
            batch_y = y_train_tensor[batch_idx].to(device)
            batch_w = weight_train_tensor[batch_idx].to(device)

            optimizer.zero_grad()
            logits = probe(batch_x)
            loss_per_sample = criterion(logits, batch_y)
            weighted_loss = (loss_per_sample * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += float((loss_per_sample * batch_w).sum().item())
            epoch_weight += float(batch_w.sum().item())

        train_loss = epoch_loss / max(epoch_weight, 1e-8)
        val_loss = weighted_average_loss(probe, criterion, X_val_tensor, y_val_tensor, weight_val_tensor, args.batch_size, device)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(probe.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_epoch={best_epoch}")

        if epoch + 1 >= args.min_epochs and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} | best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f}")
            break

    probe.load_state_dict(best_state_dict)
    probe = probe.cpu()

    train_question_ids = sorted(set(int(qid) for qid in groups[train_pool_indices]))
    train_inner_question_ids = sorted(set(int(qid) for qid in groups[inner_train_indices]))
    val_question_ids = sorted(set(int(qid) for qid in groups[val_indices]))
    test_question_ids = sorted(set(int(qid) for qid in groups[test_indices]))
    low_entropy_train_count = int(np.sum(train_sample_weights > 1.0))

    artifact = {
        "probe": probe,
        "probe_state_dict": probe.state_dict(),
        "scaler": scaler,
        "feature_key": args.feature_key,
        "feature_dim": int(X.shape[1]),
        "label_key": args.label_key,
        "probe_type": "torch_mlp",
        "random_state": args.random_state,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "train_question_ids": train_question_ids,
        "train_inner_question_ids": train_inner_question_ids,
        "val_question_ids": val_question_ids,
        "test_question_ids": test_question_ids,
        "config": {
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight,
            "early_stopping_patience": args.early_stopping_patience,
            "min_epochs": args.min_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "low_entropy_error_final_entropy_max": args.low_entropy_error_final_entropy_max,
            "low_entropy_error_final_top1_min": args.low_entropy_error_final_top1_min,
            "low_entropy_error_weight": args.low_entropy_error_weight,
            "device": str(device),
        },
    }

    torch.save(artifact, args.output_path)
    if args.label_key == "label":
        print(
            "Low-entropy hard negative weighting | "
            f"count={low_entropy_train_count} | weight={args.low_entropy_error_weight}"
        )
    print(f"Saved PyTorch probe artifact to: {args.output_path}")
    print(f"Feature dim: {artifact['feature_dim']}")
    print(
        f"Train questions: {len(train_question_ids)} | "
        f"Train-inner questions: {len(train_inner_question_ids)} | "
        f"Val questions: {len(val_question_ids)} | Test questions: {len(test_question_ids)}"
    )
if __name__ == "__main__":
    main()
