import argparse
import json
import os
import re
import statistics

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_package.answer_registry import check_answer_correctness, get_answer_extractor
from core_package.config import MODELS, SCHEDULER

# ================= Default Configuration =================
DEFAULT_LABEL_PATH = SCHEDULER.label_path
DEFAULT_SMALL_MODEL_PATH = MODELS.small_model_path
DEFAULT_LARGE_MODEL_PATH = MODELS.large_model_path
DEFAULT_TEST_SIZE = SCHEDULER.test_size
DEFAULT_RANDOM_STATE = SCHEDULER.random_state
DEFAULT_FEATURE_KEY = SCHEDULER.feature_key
DEFAULT_MLP_HIDDEN_LAYERS = SCHEDULER.mlp_hidden_layers
DEFAULT_MLP_MAX_ITER = SCHEDULER.mlp_max_iter
DEFAULT_MLP_ALPHA = SCHEDULER.mlp_alpha
DEFAULT_MLP_LEARNING_RATE_INIT = SCHEDULER.mlp_learning_rate_init
DEFAULT_THRESHOLDS = SCHEDULER.thresholds
DEFAULT_MAX_NEW_TOKENS = SCHEDULER.max_new_tokens
DEFAULT_MIN_CHUNK_TOKENS = SCHEDULER.min_chunk_tokens
DEFAULT_MAX_CHUNK_TOKENS = SCHEDULER.max_chunk_tokens
DEFAULT_TAIL_BONUS_WEIGHT = SCHEDULER.tail_bonus_weight
DEFAULT_MAX_HANDOFFS = SCHEDULER.max_handoffs
DEFAULT_LARGE_HANDOFF_CHUNKS = SCHEDULER.large_handoff_chunks
DEFAULT_PROBE_ARTIFACT_PATH = SCHEDULER.probe_artifact_path
DEFAULT_SYSTEM_PROMPT = MODELS.system_prompt
DEFAULT_BOXED_SYSTEM_PROMPT = MODELS.boxed_math_system_prompt
DEFAULT_ANSWER_TYPE = "legacy_math"
PUNCTUATIONS = [".", ",", "!", "?", "\n"]
# ========================================================


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
    parser = argparse.ArgumentParser(description="Simulate an observe-and-rollback chunk scheduler.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled chunk data.")
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Optional separate evaluation dataset path. Can be a strict labeled .pt or a raw trajectory .pt from build_dataset.",
    )
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--feature-key", default=DEFAULT_FEATURE_KEY, help="Chunk feature used by the probe.")
    parser.add_argument("--probe-artifact-path", default=DEFAULT_PROBE_ARTIFACT_PATH, help="Optional fixed probe artifact to load for fair comparisons.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Held-out question ratio.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for the split.")
    parser.add_argument("--mlp-hidden-layers", default=DEFAULT_MLP_HIDDEN_LAYERS, help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--mlp-max-iter", type=int, default=DEFAULT_MLP_MAX_ITER, help="MLP max iterations.")
    parser.add_argument("--mlp-alpha", type=float, default=DEFAULT_MLP_ALPHA, help="MLP L2 regularization.")
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=DEFAULT_MLP_LEARNING_RATE_INIT,
        help="MLP initial learning rate.",
    )
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated risk thresholds to simulate.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max total generation tokens.")
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS, help="Min tokens before punctuation can end a chunk.")
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS, help="Forced chunk cut length.")
    parser.add_argument("--tail-bonus-weight", type=float, default=DEFAULT_TAIL_BONUS_WEIGHT, help="Add alpha * generation_progress to risk score.")
    parser.add_argument("--max-handoffs", type=int, default=DEFAULT_MAX_HANDOFFS, help="Maximum number of large-model interventions.")
    parser.add_argument("--large-handoff-chunks", type=int, default=DEFAULT_LARGE_HANDOFF_CHUNKS, help="How many chunks large model handles per intervention.")
    parser.add_argument("--cooldown-chunks", type=int, default=SCHEDULER.cooldown_chunks, help="How many accepted small-model chunks to wait before another rollback handoff is allowed.")
    parser.add_argument(
        "--require-consecutive-risk",
        action="store_true",
        help="Require two consecutive risky chunks before triggering handoff. Off by default so the old single-chunk trigger can be recovered by simply omitting this flag.",
    )
    parser.add_argument("--num-test-questions", type=int, default=None, help="Optional cap on held-out test questions.")
    parser.add_argument("--trace-question-id", type=int, default=None, help="Optional question_id to print a detailed chunk routing trace for.")
    parser.add_argument("--trace-export-path", default=None, help="Optional JSON path to export per-question routing traces.")
    parser.add_argument(
        "--small-baseline-path",
        default=None,
        help="Optional JSON produced by evaluation/evaluate_model_only_accuracy.py to override stored small-model correctness.",
    )
    parser.add_argument("--answer-type", default=DEFAULT_ANSWER_TYPE, help="Answer protocol used for extraction and correctness.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt used for both small and large model generation.")
    return parser.parse_args()


def resolve_system_prompt(answer_type: str, system_prompt: str) -> str:
    if answer_type == "boxed" and system_prompt == DEFAULT_SYSTEM_PROMPT:
        return DEFAULT_BOXED_SYSTEM_PROMPT
    return system_prompt


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
        has_cue = bool(re.search(r"(therefore|thus|so|answer|final answer|total)", lower_text))
        return np.asarray([1.0 if has_cue else 0.0], dtype=np.float32)
    raise KeyError(f"Unsupported derived scalar feature: {token}")


def is_derived_feature_token(token):
    return token in {"delta_prev", "abs_delta_prev"} or token in DERIVED_SCALAR_FEATURES


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


def parse_csv_floats(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


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


def compute_token_confidence(logits):
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())
    top_values = torch.topk(probabilities, k=2)
    top1_prob = float(top_values.values[0].item())
    top2_prob = float(top_values.values[1].item()) if top_values.values.numel() > 1 else 0.0
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "margin": float(top1_prob - top2_prob),
    }


def summarize_confidence(token_confidences):
    if not token_confidences:
        zero = torch.tensor([0.0], dtype=torch.float32)
        return {
            "mean_entropy": zero.clone(),
            "max_entropy": zero.clone(),
            "final_entropy": zero.clone(),
            "mean_top1_prob": zero.clone(),
            "min_top1_prob": zero.clone(),
            "final_top1_prob": zero.clone(),
            "mean_margin": zero.clone(),
            "min_margin": zero.clone(),
            "final_margin": zero.clone(),
        }

    entropy_values = np.asarray([item["entropy"] for item in token_confidences], dtype=np.float32)
    top1_values = np.asarray([item["top1_prob"] for item in token_confidences], dtype=np.float32)
    margin_values = np.asarray([item["margin"] for item in token_confidences], dtype=np.float32)

    def scalar_tensor(value):
        return torch.tensor([float(value)], dtype=torch.float32)

    return {
        "mean_entropy": scalar_tensor(np.mean(entropy_values)),
        "max_entropy": scalar_tensor(np.max(entropy_values)),
        "final_entropy": scalar_tensor(entropy_values[-1]),
        "mean_top1_prob": scalar_tensor(np.mean(top1_values)),
        "min_top1_prob": scalar_tensor(np.min(top1_values)),
        "final_top1_prob": scalar_tensor(top1_values[-1]),
        "mean_margin": scalar_tensor(np.mean(margin_values)),
        "min_margin": scalar_tensor(np.min(margin_values)),
        "final_margin": scalar_tensor(margin_values[-1]),
    }


def build_generation_messages(question, assistant_prefix=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if assistant_prefix is not None:
        messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def build_generation_inputs(tokenizer, question, assistant_prefix, system_prompt=DEFAULT_SYSTEM_PROMPT):
    normalized_prefix = None
    if assistant_prefix is not None:
        normalized_prefix = assistant_prefix.rstrip()
        if not normalized_prefix:
            normalized_prefix = None

    messages = build_generation_messages(question, assistant_prefix=normalized_prefix, system_prompt=system_prompt)
    if normalized_prefix is None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ), normalized_prefix

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        continue_final_message=True,
    ), normalized_prefix


def prompt_token_count(tokenizer, question, system_prompt):
    inputs, _ = build_generation_inputs(tokenizer, question, assistant_prefix=None, system_prompt=system_prompt)
    return int(inputs["input_ids"].shape[1])

def decode_tokens(tokenizer, token_ids):
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_question_records(dataset, feature_key):
    question_records = {}
    required_tokens = parse_feature_spec(feature_key)

    if dataset and isinstance(dataset[0], dict) and "chunks" in dataset[0]:
        for item in dataset:
            question_id = int(item["question_id"])
            chunks = []
            has_all_components = True
            for chunk in item.get("chunks", []):
                for token in required_tokens:
                    if is_derived_feature_token(token):
                        continue
                    resolved_key = canonical_feature_name(token)
                    if resolved_key not in chunk:
                        has_all_components = False
                        break
                if not has_all_components:
                    break
                chunks.append(chunk)
            if not has_all_components or not chunks:
                continue

            question_records[question_id] = {
                "question_id": question_id,
                "question": item["question"],
                "ground_truth_final_answer": item.get("ground_truth_final_answer"),
                "small_final_answer": item.get("model_final_answer"),
                "small_is_correct": bool(item.get("is_final_correct", False)),
                "chunks": sorted(chunks, key=lambda chunk: int(chunk["chunk_id"])),
            }
        return question_records

    for item in dataset:
        has_all_components = True
        for token in required_tokens:
            if is_derived_feature_token(token):
                continue
            resolved_key = canonical_feature_name(token)
            if resolved_key not in item:
                has_all_components = False
                break
        if not has_all_components:
            continue
        question_id = int(item["question_id"])
        record = question_records.setdefault(
            question_id,
            {
                "question_id": question_id,
                "question": item["question"],
                "ground_truth_final_answer": item["ground_truth_final_answer"],
                "small_final_answer": item["model_final_answer"],
                "small_is_correct": bool(item["is_final_correct"]),
                "chunks": [],
            },
        )
        record["chunks"].append(item)

    for record in question_records.values():
        record["chunks"] = sorted(record["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    return question_records


def apply_small_baseline_overrides(question_records, baseline_path):
    if not baseline_path or not os.path.exists(baseline_path):
        return 0

    with open(baseline_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("rows", [])
    updated = 0
    for row in rows:
        qid = int(row["question_id"])
        if qid not in question_records:
            continue
        question_records[qid]["small_final_answer"] = row.get("pred_final_answer")
        question_records[qid]["small_is_correct"] = bool(row.get("is_correct", False))
        updated += 1
    return updated


def build_feature_arrays(question_records, feature_key):
    rows = []
    labels = []
    groups = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            if "label" not in chunk:
                raise KeyError("Chunk labels are required for probe fitting but missing in the provided training dataset.")
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(build_feature_vector(chunk, prev_chunk, total_chunks, feature_key))
            labels.append(int(chunk["label"]))
            groups.append(question_id)
    return np.stack(rows), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def fit_probe(question_records, args):
    X, y, groups = build_feature_arrays(question_records, args.feature_key)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_indices])
    y_train = y[train_indices]

    probe = build_probe(args)
    X_fit, y_fit = upsample_minority_class(X_train, y_train, args.random_state)
    probe.fit(X_fit, y_fit)

    train_question_ids = sorted(set(int(qid) for qid in groups[train_indices]))
    test_question_ids = sorted(set(int(qid) for qid in groups[test_indices]))
    return probe, scaler, train_question_ids, test_question_ids


def artifact_trigger_score(artifact, positive_prob):
    label_key = artifact.get("label_key", "label")
    if label_key == "takeover_beneficial":
        return positive_prob
    return 1.0 - positive_prob


def load_probe_artifact(args):
    if not args.probe_artifact_path or not os.path.exists(args.probe_artifact_path):
        return None
    artifact = torch.load(args.probe_artifact_path)
    probe = artifact.get("probe")
    if probe is None and artifact.get("probe_state_dict") is not None:
        hidden_layers_text = artifact["config"]["hidden_layers"]
        hidden_layers = tuple(int(part.strip()) for part in hidden_layers_text.split(",") if part.strip())
        dropout = float(artifact.get("config", {}).get("dropout", 0.0))
        probe = TorchMLPProbe(
            input_dim=int(artifact["feature_dim"]),
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        probe.load_state_dict(artifact["probe_state_dict"])
        artifact["probe"] = probe
    return artifact


def run_chunk(model, tokenizer, question, assistant_prefix, max_new_tokens, min_chunk_tokens, max_chunk_tokens, system_prompt=DEFAULT_SYSTEM_PROMPT):
    inputs, normalized_prefix = build_generation_inputs(tokenizer, question, assistant_prefix, system_prompt=system_prompt)
    input_ids = inputs.input_ids.to(model.device)
    past_key_values = None

    generated_token_ids = []
    chunk_token_ids = []
    chunk_hidden_states = []
    chunk_confidences = []
    reached_eos = False
    cut_reason = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

        if chunk_token_ids:
            chunk_hidden_states.append(outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu())

        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        chunk_confidences.append(compute_token_confidence(logits))
        next_id = torch.argmax(logits).item()
        generated_token_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            reached_eos = True
            break

        chunk_token_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        chunk_len = len(chunk_token_ids)
        hit_punctuation = any(p in token_text for p in PUNCTUATIONS)

        if hit_punctuation and chunk_len >= min_chunk_tokens:
            cut_reason = "punctuation"
        elif chunk_len >= max_chunk_tokens:
            cut_reason = "max_tokens"

        if cut_reason is not None:
            with torch.no_grad():
                final_outputs = model(
                    input_ids=torch.tensor([[next_id]], device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
            chunk_hidden_states.append(final_outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu())
            break

        input_ids = torch.tensor([[next_id]], device=model.device)

    if chunk_token_ids and len(chunk_hidden_states) < len(chunk_token_ids):
        last_token_id = chunk_token_ids[-1]
        with torch.no_grad():
            final_outputs = model(
                input_ids=torch.tensor([[last_token_id]], device=model.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        chunk_hidden_states.append(final_outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu())

    chunk_text = decode_tokens(tokenizer, chunk_token_ids).strip()
    full_reasoning = (normalized_prefix or "") + chunk_text
    boundary_hidden_state = chunk_hidden_states[-1] if chunk_hidden_states else None
    mean_hidden_state = torch.stack(chunk_hidden_states).mean(dim=0) if chunk_hidden_states else None

    if cut_reason is None and chunk_token_ids:
        cut_reason = "tail"

    return {
        "full_reasoning": full_reasoning,
        "chunk_text": chunk_text,
        "token_ids": chunk_token_ids,
        "generated_token_count": len(chunk_token_ids),
        "boundary_hidden_state": boundary_hidden_state,
        "mean_hidden_state": mean_hidden_state,
        "cut_reason": cut_reason,
        "reached_eos": reached_eos,
        **summarize_confidence(chunk_confidences),
    }


def run_large_handoff(model, tokenizer, question, assistant_prefix, args):
    prefix = assistant_prefix
    total_generated_tokens = 0
    generated_chunks = 0
    reached_eos = False
    chunks = []

    for handoff_chunk_idx in range(args.large_handoff_chunks):
        remaining_budget = max(args.max_new_tokens - total_generated_tokens, 1)
        chunk_result = run_chunk(
            model=model,
            tokenizer=tokenizer,
            question=question,
            assistant_prefix=prefix,
            max_new_tokens=remaining_budget,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
            system_prompt=args.system_prompt,
        )
        prefix = chunk_result["full_reasoning"]
        total_generated_tokens += chunk_result["generated_token_count"]
        generated_chunks += 1
        reached_eos = chunk_result["reached_eos"]
        chunks.append(
            {
                "handoff_local_chunk_id": handoff_chunk_idx,
                "chunk_text": chunk_result["chunk_text"],
                "generated_token_count": chunk_result["generated_token_count"],
                "cut_reason": chunk_result["cut_reason"],
                "reached_eos": chunk_result["reached_eos"],
            }
        )
        if reached_eos or chunk_result["generated_token_count"] == 0:
            break

    return {
        "full_reasoning": prefix,
        "generated_token_count": total_generated_tokens,
        "generated_chunks": generated_chunks,
        "reached_eos": reached_eos,
        "chunks": chunks,
    }


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def simulate_question(record, small_model, small_tokenizer, large_model, large_tokenizer, probe, scaler, threshold, args, artifact=None):
    question = record["question"]
    ground_truth_final_answer = record["ground_truth_final_answer"]
    answer_extractor = get_answer_extractor(args.answer_type)
    question_prompt_token_count = prompt_token_count(small_tokenizer, question, args.system_prompt)
    prefix = None
    total_tokens = 0
    total_large_tokens = 0
    handoff_count = 0
    chunk_index = 0
    triggered = False
    trigger_scores = []
    trigger_progresses = []
    runtime_small_chunks = []
    reset_prev_chunk = False
    cooldown_remaining = 0
    route_trace = []
    previous_combined_score = None

    while total_tokens < args.max_new_tokens:
        safe_prefix = prefix
        remaining_budget = max(args.max_new_tokens - total_tokens, 1)
        small_chunk = run_chunk(
            model=small_model,
            tokenizer=small_tokenizer,
            question=question,
            assistant_prefix=prefix,
            max_new_tokens=remaining_budget,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
            system_prompt=args.system_prompt,
        )

        if small_chunk["generated_token_count"] == 0:
            prefix = safe_prefix
            break

        total_tokens += small_chunk["generated_token_count"]
        progress_ratio = total_tokens / max(args.max_new_tokens, 1)
        small_chunk["chunk_id"] = chunk_index
        runtime_small_chunks.append(small_chunk)

        if reset_prev_chunk:
            prev_chunk = None
            reset_prev_chunk = False
        else:
            prev_chunk = runtime_small_chunks[-2] if len(runtime_small_chunks) >= 2 else None
        total_chunks_for_features = max(len(runtime_small_chunks), int(small_chunk["chunk_id"]) + 1)
        feature_spec = artifact.get("feature_key", args.feature_key) if artifact is not None else args.feature_key
        feature_vector = build_feature_vector(small_chunk, prev_chunk, total_chunks_for_features, feature_spec)
        features = scaler.transform([feature_vector])
        positive_prob = probe.predict_proba(features)[0, 1]
        trigger_score = artifact_trigger_score(artifact, positive_prob) if artifact is not None else (1.0 - positive_prob)
        combined_score = trigger_score + args.tail_bonus_weight * progress_ratio
        prev_score_for_trace = previous_combined_score

        # SVAMP false-alarm mitigation:
        # When enabled, we only trigger after two consecutive risky chunks.
        # This keeps the old behavior fully recoverable: do not pass
        # --require-consecutive-risk and the scheduler falls back to the
        # original single-chunk trigger rule.
        meets_risk_trigger = combined_score >= threshold
        if args.require_consecutive_risk:
            meets_risk_trigger = (
                combined_score >= threshold
                and previous_combined_score is not None
                and previous_combined_score >= threshold
            )

        if meets_risk_trigger and handoff_count < args.max_handoffs and cooldown_remaining <= 0:
            triggered = True
            trigger_scores.append(combined_score)
            trigger_progresses.append(progress_ratio)

            route_trace.append(
                {
                    "event": "small_observe_rollback",
                    "chunk_id": int(small_chunk["chunk_id"]),
                    "chunk_text": small_chunk["chunk_text"],
                    "generated_token_count": small_chunk["generated_token_count"],
                    "trigger_score": float(trigger_score),
                    "combined_score": float(combined_score),
                    "previous_combined_score": None if prev_score_for_trace is None else float(prev_score_for_trace),
                    "progress_ratio": float(progress_ratio),
                    "cut_reason": small_chunk["cut_reason"],
                    "trigger_rule": "consecutive_two_chunk_risk" if args.require_consecutive_risk else "single_chunk_risk",
                }
            )

            # Roll back the discarded small-model chunk before applying the large-model handoff.
            total_tokens -= small_chunk["generated_token_count"]
            runtime_small_chunks.pop()

            large_result = run_large_handoff(
                model=large_model,
                tokenizer=large_tokenizer,
                question=question,
                assistant_prefix=safe_prefix,
                args=args,
            )
            prefix = large_result["full_reasoning"]
            total_tokens += large_result["generated_token_count"]
            total_large_tokens += large_result["generated_token_count"]
            handoff_count += 1
            route_trace.append(
                {
                    "event": "large_handoff",
                    "handoff_index": handoff_count,
                    "generated_token_count": large_result["generated_token_count"],
                    "generated_chunks": large_result["generated_chunks"],
                    "chunks": large_result["chunks"],
                }
            )
            chunk_index += large_result["generated_chunks"]
            reset_prev_chunk = True
            cooldown_remaining = args.cooldown_chunks
            previous_combined_score = None
            if large_result["reached_eos"]:
                break
            continue

        route_trace.append(
            {
                "event": "small_accept",
                "chunk_id": int(small_chunk["chunk_id"]),
                "chunk_text": small_chunk["chunk_text"],
                "generated_token_count": small_chunk["generated_token_count"],
                "trigger_score": float(trigger_score),
                "combined_score": float(combined_score),
                "previous_combined_score": None if prev_score_for_trace is None else float(prev_score_for_trace),
                "progress_ratio": float(progress_ratio),
                "cut_reason": small_chunk["cut_reason"],
            }
        )
        prefix = small_chunk["full_reasoning"]
        chunk_index += 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        previous_combined_score = combined_score
        if small_chunk["reached_eos"]:
            break

    # Evaluation hygiene fix:
    # If no handoff happened, the scheduler should exactly inherit the stored
    # small-model baseline instead of being judged on chunk-by-chunk decoding
    # drift. This keeps "no trigger" rows from looking like fake harms/rescues.
    if handoff_count == 0:
        final_reasoning = prefix or ""
        final_answer = record.get("small_final_answer")
        scheduled_is_correct = bool(record.get("small_is_correct", False))
    else:
        final_reasoning = prefix or ""
        final_answer, has_answer = answer_extractor(final_reasoning)
        scheduled_is_correct = has_answer and check_answer_correctness(
            final_answer,
            ground_truth_final_answer,
            args.answer_type,
        )
    return {
        "scheduled_is_correct": scheduled_is_correct,
        "scheduled_final_answer": final_answer,
        "full_reasoning": final_reasoning,
        "prompt_token_count": question_prompt_token_count,
        "triggered": triggered,
        "handoff_count": handoff_count,
        "large_generated_tokens": total_large_tokens,
        "avg_trigger_score": safe_mean(trigger_scores),
        "avg_trigger_progress": safe_mean(trigger_progresses),
        "route_trace": route_trace,
    }


def simulate_threshold(test_records, small_model, small_tokenizer, large_model, large_tokenizer, probe, scaler, threshold, args, artifact=None):
    small_only_correct = 0
    scheduled_correct = 0
    error_questions = 0
    triggered_questions = 0
    triggered_wrong_questions = 0
    false_alarm_correct_questions = 0
    handoff_counts = []
    large_takeover_tokens = []
    trigger_progresses = []
    per_question_rows = []

    progress = tqdm(test_records, desc=f"Threshold {threshold:.2f}", leave=False)
    for record in progress:
        small_is_correct = record["small_is_correct"]
        if small_is_correct:
            small_only_correct += 1
        else:
            error_questions += 1

        result = simulate_question(
            record=record,
            small_model=small_model,
            small_tokenizer=small_tokenizer,
            large_model=large_model,
            large_tokenizer=large_tokenizer,
            probe=probe,
            scaler=scaler,
            threshold=threshold,
            args=args,
            artifact=artifact,
        )

        scheduled_correct += int(result["scheduled_is_correct"])
        if result["triggered"]:
            triggered_questions += 1
            handoff_counts.append(result["handoff_count"])
            large_takeover_tokens.append(result["large_generated_tokens"])
            if not np.isnan(result["avg_trigger_progress"]):
                trigger_progresses.append(result["avg_trigger_progress"])
            if not small_is_correct:
                triggered_wrong_questions += 1
            else:
                false_alarm_correct_questions += 1

        per_question_rows.append(
            {
                "question_id": record["question_id"],
                "prompt_token_count": result["prompt_token_count"],
                "small_is_correct": small_is_correct,
                "scheduled_is_correct": result["scheduled_is_correct"],
                "triggered": result["triggered"],
                "handoff_count": result["handoff_count"],
                "avg_trigger_score": result["avg_trigger_score"],
                "avg_trigger_progress": result["avg_trigger_progress"],
                "small_final_answer": record["small_final_answer"],
                "scheduled_final_answer": result["scheduled_final_answer"],
                "route_trace": result["route_trace"],
            }
        )

    total_questions = len(test_records)
    correct_questions = total_questions - error_questions
    return {
        "threshold": threshold,
        "tail_bonus_weight": args.tail_bonus_weight,
        "questions_total": total_questions,
        "small_only_accuracy": small_only_correct / total_questions,
        "scheduled_accuracy": scheduled_correct / total_questions,
        "scheduled_gain_over_small": (scheduled_correct - small_only_correct) / total_questions,
        "trigger_rate": triggered_questions / total_questions,
        "questions_triggered": triggered_questions,
        "error_questions_total": error_questions,
        "error_questions_triggered": triggered_wrong_questions,
        "false_alarm_correct_question_rate": false_alarm_correct_questions / correct_questions if correct_questions else float("nan"),
        "avg_handoff_count": safe_mean(handoff_counts),
        "avg_large_takeover_tokens": safe_mean(large_takeover_tokens),
        "avg_trigger_progress": safe_mean(trigger_progresses),
        "per_question_rows": per_question_rows,
    }


def print_route_trace(row):
    print("\nRoute trace")
    print("=" * 50)
    print(
        f"question_id={row['question_id']} | "
        f"small_correct={row['small_is_correct']} | "
        f"scheduled_correct={row['scheduled_is_correct']} | "
        f"handoff_count={row['handoff_count']}"
    )
    for step_idx, event in enumerate(row.get("route_trace", []), start=1):
        if event["event"] == "small_accept":
            print(
                f"[{step_idx}] SMALL accept chunk#{event['chunk_id']} | "
                f"score={event['combined_score']:.4f} | text={event['chunk_text']!r}"
            )
        elif event["event"] == "small_observe_rollback":
            print(
                f"[{step_idx}] SMALL rollback chunk#{event['chunk_id']} | "
                f"score={event['combined_score']:.4f} | text={event['chunk_text']!r}"
            )
        elif event["event"] == "large_handoff":
            print(
                f"[{step_idx}] LARGE handoff#{event['handoff_index']} | "
                f"chunks={event['generated_chunks']} | tokens={event['generated_token_count']}"
            )
            for chunk in event.get("chunks", []):
                print(
                    f"      - large_chunk#{chunk['handoff_local_chunk_id']} | "
                    f"text={chunk['chunk_text']!r}"
                )


def print_summary(summary):
    print("\nObserve-and-rollback scheduler simulation")
    print("=" * 50)
    print(f"Threshold: {summary['threshold']:.2f}")
    print(f"Tail bonus weight: {summary['tail_bonus_weight']:.4f}")
    print(f"Questions total: {summary['questions_total']}")
    print(f"Small-only accuracy: {summary['small_only_accuracy']:.4f}")
    print(f"Scheduled accuracy: {summary['scheduled_accuracy']:.4f}")
    print(f"Scheduled gain over small: {summary['scheduled_gain_over_small']:+.4f}")
    print(f"Trigger rate: {summary['trigger_rate']:.4f} ({summary['questions_triggered']}/{summary['questions_total']})")
    print(f"Error questions total: {summary['error_questions_total']}")
    print(f"Error questions triggered: {summary['error_questions_triggered']}")
    print(f"False-alarm correct-question rate: {summary['false_alarm_correct_question_rate']:.4f}")
    print(f"Avg handoff count: {summary['avg_handoff_count']:.2f}")
    print(f"Avg trigger progress: {summary['avg_trigger_progress']:.4f}")
    print(f"Avg large takeover tokens: {summary['avg_large_takeover_tokens']:.2f}")


def main():
    args = parse_args()
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)
    thresholds = parse_csv_floats(args.thresholds)

    print(f"Loading training dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)

    requested_feature_key = args.feature_key
    artifact = load_probe_artifact(args)
    if artifact is not None:
        args.feature_key = artifact.get("feature_key", args.feature_key)
        if requested_feature_key != args.feature_key:
            print(f"Using artifact feature spec '{args.feature_key}' instead of requested '{requested_feature_key}'.")

    question_records = build_question_records(dataset, args.feature_key)
    updated_small_records = apply_small_baseline_overrides(question_records, args.small_baseline_path)
    print(f"Loaded training questions with {args.feature_key}: {len(question_records)}")
    if updated_small_records:
        print(f"Overrode stored small-model baseline for {updated_small_records} training questions from: {args.small_baseline_path}")

    eval_question_records = question_records
    if args.eval_data_path is not None:
        print(f"Loading separate evaluation dataset from: {args.eval_data_path}")
        eval_dataset = torch.load(args.eval_data_path)
        eval_question_records = build_question_records(eval_dataset, args.feature_key)
        updated_eval_small_records = apply_small_baseline_overrides(eval_question_records, args.small_baseline_path)
        print(f"Loaded eval questions with {args.feature_key}: {len(eval_question_records)}")
        if updated_eval_small_records:
            print(f"Overrode stored small-model baseline for {updated_eval_small_records} eval questions from: {args.small_baseline_path}")
    if artifact is not None:
        print(f"Loading fixed probe artifact from: {args.probe_artifact_path}")
        probe = artifact.get("probe")
        scaler = artifact["scaler"]
        train_question_ids = list(artifact["train_question_ids"])
        if args.eval_data_path is not None:
            test_question_ids = sorted(eval_question_records.keys())
        else:
            test_question_ids = list(artifact["test_question_ids"])
    else:
        raise FileNotFoundError(
            "Multi-handoff scheduler now requires a trained takeover_beneficial artifact. "
            "Please run core_package/probes/train_probe_artifact_torch.py with the desired label key first."
        )
    print(f"Train questions: {len(train_question_ids)} | Test questions: {len(test_question_ids)}")

    test_records = [eval_question_records[question_id] for question_id in sorted(test_question_ids) if question_id in eval_question_records]
    if args.num_test_questions is not None:
        test_records = test_records[: args.num_test_questions]
        print(f"Using first {len(test_records)} held-out questions for multi hand-off simulation.")

    print(f"Loading small model from: {args.small_model_path}")
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    print(f"Loading large model from: {args.large_model_path}")
    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, local_files_only=True)
    large_model = AutoModelForCausalLM.from_pretrained(
        args.large_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    large_model.eval()

    print(f"Simulating thresholds: {thresholds}")
    summaries = []
    for threshold in thresholds:
        summary = simulate_threshold(
            test_records=test_records,
            small_model=small_model,
            small_tokenizer=small_tokenizer,
            large_model=large_model,
            large_tokenizer=large_tokenizer,
            probe=probe,
            scaler=scaler,
            threshold=threshold,
            args=args,
            artifact=artifact,
        )
        summaries.append(summary)
        print_summary(summary)

    if args.trace_export_path:
        export_rows = []
        for summary in summaries:
            export_rows.append(
                {
                    "threshold": summary["threshold"],
                    "tail_bonus_weight": summary["tail_bonus_weight"],
                    "per_question_rows": summary["per_question_rows"],
                }
            )
        with open(args.trace_export_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(export_rows), f, ensure_ascii=False, indent=2)
        print(f"Saved routing traces to: {args.trace_export_path}")

    if args.trace_question_id is not None:
        for summary in summaries:
            matched = next(
                (row for row in summary["per_question_rows"] if int(row["question_id"]) == int(args.trace_question_id)),
                None,
            )
            if matched is not None:
                print(f"\nDetailed trace for threshold={summary['threshold']:.2f}")
                print_route_trace(matched)
                break

    best_summary = max(summaries, key=lambda item: item["scheduled_gain_over_small"])
    print("\nRecommended observe-and-rollback threshold")
    print("=" * 50)
    print(
        f"threshold={best_summary['threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['scheduled_gain_over_small']:+.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()


