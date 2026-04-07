import argparse
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

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_SMALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B")
DEFAULT_LARGE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 55
DEFAULT_FEATURE_KEY = "boundary"
DEFAULT_MLP_HIDDEN_LAYERS = "512,128,32"
DEFAULT_MLP_MAX_ITER = 300
DEFAULT_MLP_ALPHA = 1e-4
DEFAULT_MLP_LEARNING_RATE_INIT = 1e-3
DEFAULT_THRESHOLDS = "0.15,0.20,0.25"
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_MIN_CHUNK_TOKENS = 5
DEFAULT_MAX_CHUNK_TOKENS = 30
DEFAULT_TAIL_BONUS_WEIGHT = 0.0
DEFAULT_MAX_HANDOFFS = 2
DEFAULT_LARGE_HANDOFF_CHUNKS = 2
DEFAULT_PROBE_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "beneficial_probe_artifact.pt")
DEFAULT_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step."
PUNCTUATIONS = [".", ",", "!", "?", "\n"]
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate a multi hand-off small/large scheduler.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled chunk data.")
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
    parser.add_argument("--num-test-questions", type=int, default=None, help="Optional cap on held-out test questions.")
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


def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_numeric_text(text):
    return text.replace(",", "").strip().rstrip(".")


def extract_final_answer(text):
    if not text:
        return None

    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_matches:
        boxed_value = extract_last_number(boxed_matches[-1])
        if boxed_value is not None:
            return boxed_value

    explicit_patterns = [
        r"(?i)final answer\s*[:?]\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n]+)",
        r"(?i)answer\s*[:?]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            explicit_value = extract_last_number(matches[-1])
            if explicit_value is not None:
                return explicit_value

    return extract_last_number(normalize_numeric_text(text))


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


def build_generation_messages(question, assistant_prefix=None):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if assistant_prefix is not None:
        messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def build_generation_inputs(tokenizer, question, assistant_prefix):
    normalized_prefix = None
    if assistant_prefix is not None:
        normalized_prefix = assistant_prefix.rstrip()
        if not normalized_prefix:
            normalized_prefix = None

    messages = build_generation_messages(question, assistant_prefix=normalized_prefix)
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


def decode_tokens(tokenizer, token_ids):
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_question_records(dataset, feature_key):
    question_records = {}
    required_tokens = parse_feature_spec(feature_key)
    for item in dataset:
        has_all_components = True
        for token in required_tokens:
            if token in {"delta_prev", "abs_delta_prev", "relative_position", "remaining_ratio"}:
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


def build_feature_arrays(question_records, feature_key):
    rows = []
    labels = []
    groups = []
    for question_id, record in question_records.items():
        chunks = record["chunks"]
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
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
    if artifact.get("label_key") != "takeover_beneficial":
        raise ValueError(
            f"Multi-handoff scheduler expects a takeover_beneficial artifact, but got label_key={artifact.get('label_key')!r} from {args.probe_artifact_path}."
        )
    return artifact


def run_chunk(model, tokenizer, question, assistant_prefix, max_new_tokens, min_chunk_tokens, max_chunk_tokens):
    inputs, normalized_prefix = build_generation_inputs(tokenizer, question, assistant_prefix)
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

    for _ in range(args.large_handoff_chunks):
        remaining_budget = max(args.max_new_tokens - total_generated_tokens, 1)
        chunk_result = run_chunk(
            model=model,
            tokenizer=tokenizer,
            question=question,
            assistant_prefix=prefix,
            max_new_tokens=remaining_budget,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
        )
        prefix = chunk_result["full_reasoning"]
        total_generated_tokens += chunk_result["generated_token_count"]
        generated_chunks += 1
        reached_eos = chunk_result["reached_eos"]
        if reached_eos or chunk_result["generated_token_count"] == 0:
            break

    return {
        "full_reasoning": prefix,
        "generated_token_count": total_generated_tokens,
        "generated_chunks": generated_chunks,
        "reached_eos": reached_eos,
    }


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def simulate_question(record, small_model, small_tokenizer, large_model, large_tokenizer, probe, scaler, threshold, args, artifact=None):
    question = record["question"]
    ground_truth_final_answer = record["ground_truth_final_answer"]
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

        if combined_score >= threshold and handoff_count < args.max_handoffs:
            triggered = True
            trigger_scores.append(combined_score)
            trigger_progresses.append(progress_ratio)

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
            chunk_index += large_result["generated_chunks"]
            reset_prev_chunk = True
            if large_result["reached_eos"]:
                break
            continue

        prefix = small_chunk["full_reasoning"]
        chunk_index += 1
        if small_chunk["reached_eos"]:
            break

    final_reasoning = prefix or ""
    final_answer = extract_final_answer(final_reasoning)
    scheduled_is_correct = final_answer == ground_truth_final_answer
    return {
        "scheduled_is_correct": scheduled_is_correct,
        "scheduled_final_answer": final_answer,
        "full_reasoning": final_reasoning,
        "triggered": triggered,
        "handoff_count": handoff_count,
        "large_generated_tokens": total_large_tokens,
        "avg_trigger_score": safe_mean(trigger_scores),
        "avg_trigger_progress": safe_mean(trigger_progresses),
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
                "small_is_correct": small_is_correct,
                "scheduled_is_correct": result["scheduled_is_correct"],
                "triggered": result["triggered"],
                "handoff_count": result["handoff_count"],
                "avg_trigger_score": result["avg_trigger_score"],
                "avg_trigger_progress": result["avg_trigger_progress"],
                "small_final_answer": record["small_final_answer"],
                "scheduled_final_answer": result["scheduled_final_answer"],
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


def print_summary(summary):
    print("\nMulti hand-off scheduler simulation")
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
    thresholds = parse_csv_floats(args.thresholds)

    print(f"Loading labeled chunk dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)

    requested_feature_key = args.feature_key
    artifact = load_probe_artifact(args)
    if artifact is not None:
        args.feature_key = artifact.get("feature_key", args.feature_key)
        if requested_feature_key != args.feature_key:
            print(f"Using artifact feature spec '{args.feature_key}' instead of requested '{requested_feature_key}'.")

    question_records = build_question_records(dataset, args.feature_key)
    print(f"Loaded questions with {args.feature_key}: {len(question_records)}")
    if artifact is not None:
        print(f"Loading fixed probe artifact from: {args.probe_artifact_path}")
        probe = artifact["probe"]
        scaler = artifact["scaler"]
        train_question_ids = list(artifact["train_question_ids"])
        test_question_ids = list(artifact["test_question_ids"])
    else:
        raise FileNotFoundError(
            "Multi-handoff scheduler now requires a trained takeover_beneficial artifact. "
            "Please run probes/train_probe_artifact.py with --label-key takeover_beneficial first."
        )
    print(f"Train questions: {len(train_question_ids)} | Test questions: {len(test_question_ids)}")

    test_records = [question_records[question_id] for question_id in sorted(test_question_ids)]
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

    best_summary = max(summaries, key=lambda item: item["scheduled_gain_over_small"])
    print("\nRecommended multi hand-off threshold")
    print("=" * 50)
    print(
        f"threshold={best_summary['threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['scheduled_gain_over_small']:+.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()

