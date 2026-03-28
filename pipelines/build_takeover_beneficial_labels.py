import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "gsm8k_takeover_beneficial_labels.pt")
DEFAULT_SMALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "takeover_beneficial_cache.pt")
DEFAULT_PROBE_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "probe_artifact.pt")
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_SAVE_EVERY = 25
DEFAULT_START_QUESTION = 0
DEFAULT_RANDOM_STATE = 55
DEFAULT_FEATURE_KEY = "boundary"
DEFAULT_CANDIDATE_MODE = "hybrid"
DEFAULT_TOP_K = 2
DEFAULT_EXPLORE_POSITIONS = "middle,last"
DEFAULT_MLP_HIDDEN_LAYERS = "512,128,32"
DEFAULT_MLP_MAX_ITER = 300
DEFAULT_MLP_ALPHA = 1e-4
DEFAULT_MLP_LEARNING_RATE_INIT = 1e-3
DEFAULT_MIN_CHUNK_TOKENS = 5
DEFAULT_MAX_CHUNK_TOKENS = 30
DEFAULT_LARGE_HANDOFF_CHUNKS = 2
DEFAULT_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step."
PUNCTUATIONS = [".", ",", "!", "?", "\n"]
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Build chunk-level takeover_beneficial labels.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to strict labeled chunk data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save takeover_beneficial labels.")
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Path to cached takeover generations.")
    parser.add_argument("--probe-artifact-path", default=DEFAULT_PROBE_ARTIFACT_PATH, help="Optional fixed probe artifact for candidate selection.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max total generation tokens for local multi-handoff rollouts.")
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS, help="Min tokens before punctuation can end a chunk.")
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS, help="Forced chunk cut length.")
    parser.add_argument("--large-handoff-chunks", type=int, default=DEFAULT_LARGE_HANDOFF_CHUNKS, help="How many chunks large model handles for each labeled intervention.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save every N newly processed chunks.")
    parser.add_argument("--start-question", type=int, default=DEFAULT_START_QUESTION, help="Skip questions below this id.")
    parser.add_argument("--num-questions", type=int, default=None, help="Only process the first N questions after filtering.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output and cache files.")
    parser.add_argument("--only-small-wrong", action="store_true", help="Only label candidates from questions the small model originally gets wrong.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for candidate scoring.")
    parser.add_argument("--feature-key", default=DEFAULT_FEATURE_KEY, help="Feature key used for candidate risk scoring.")
    parser.add_argument(
        "--candidate-mode",
        choices=["all", "topk", "hybrid"],
        default=DEFAULT_CANDIDATE_MODE,
        help="How to choose chunk candidates for beneficial labeling.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k risky chunks per question when using topk/hybrid.")
    parser.add_argument(
        "--explore-positions",
        default=DEFAULT_EXPLORE_POSITIONS,
        help="Comma-separated exploratory chunk positions for hybrid mode: first,middle,last.",
    )
    parser.add_argument("--mlp-hidden-layers", default=DEFAULT_MLP_HIDDEN_LAYERS, help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--mlp-max-iter", type=int, default=DEFAULT_MLP_MAX_ITER, help="MLP max iterations for candidate scoring.")
    parser.add_argument("--mlp-alpha", type=float, default=DEFAULT_MLP_ALPHA, help="MLP L2 regularization for candidate scoring.")
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=DEFAULT_MLP_LEARNING_RATE_INIT,
        help="MLP initial learning rate for candidate scoring.",
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


def parse_csv_tokens(text):
    values = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        values.append(part)
    return values


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
        r"(?i)final answer\s*[:锛歖?\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n]+)",
        r"(?i)answer\s*[:锛歖?\s*([^\n]+)",
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


def run_chunk(model, tokenizer, question, assistant_prefix, max_new_tokens, min_chunk_tokens, max_chunk_tokens):
    inputs, normalized_prefix = build_generation_inputs(tokenizer, question, assistant_prefix)
    input_ids = inputs.input_ids.to(model.device)
    past_key_values = None

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

        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        chunk_confidences.append(compute_token_confidence(logits))
        next_id = torch.argmax(logits).item()

        if next_id == tokenizer.eos_token_id:
            reached_eos = True
            break

        chunk_token_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        with torch.no_grad():
            token_outputs = model(
                input_ids=torch.tensor([[next_id]], device=model.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        past_key_values = token_outputs.past_key_values
        chunk_hidden_states.append(token_outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu())

        chunk_len = len(chunk_token_ids)
        hit_punctuation = any(p in token_text for p in PUNCTUATIONS)
        if hit_punctuation and chunk_len >= min_chunk_tokens:
            cut_reason = "punctuation"
        elif chunk_len >= max_chunk_tokens:
            cut_reason = "max_tokens"

        if cut_reason is not None:
            break

        input_ids = torch.tensor([[next_id]], device=model.device)

    chunk_text = decode_tokens(tokenizer, chunk_token_ids).strip()
    full_reasoning = (normalized_prefix or "") + chunk_text
    boundary_hidden_state = chunk_hidden_states[-1] if chunk_hidden_states else None
    mean_hidden_state = torch.stack(chunk_hidden_states).mean(dim=0) if chunk_hidden_states else None
    if cut_reason is None and chunk_token_ids:
        cut_reason = "tail"

    return {
        "full_reasoning": full_reasoning,
        "chunk_text": chunk_text,
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


def simulate_local_handoff(question, assistant_prefix, small_model, small_tokenizer, large_model, large_tokenizer, args):
    prefix = assistant_prefix
    total_generated_tokens = 0

    large_result = run_large_handoff(
        model=large_model,
        tokenizer=large_tokenizer,
        question=question,
        assistant_prefix=prefix,
        args=args,
    )
    prefix = large_result["full_reasoning"]
    total_generated_tokens += large_result["generated_token_count"]
    if large_result["reached_eos"]:
        final_reasoning = prefix or ""
        return {
            "full_reasoning": final_reasoning,
            "final_answer": extract_final_answer(final_reasoning),
            "generated_token_count": total_generated_tokens,
            "large_generated_tokens": large_result["generated_token_count"],
            "large_generated_chunks": large_result["generated_chunks"],
        }

    while total_generated_tokens < args.max_new_tokens:
        remaining_budget = max(args.max_new_tokens - total_generated_tokens, 1)
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
            break
        prefix = small_chunk["full_reasoning"]
        total_generated_tokens += small_chunk["generated_token_count"]
        if small_chunk["reached_eos"]:
            break

    final_reasoning = prefix or ""
    return {
        "full_reasoning": final_reasoning,
        "final_answer": extract_final_answer(final_reasoning),
        "generated_token_count": total_generated_tokens,
        "large_generated_tokens": large_result["generated_token_count"],
        "large_generated_chunks": large_result["generated_chunks"],
    }


def load_output(output_path, resume):
    if not resume or not os.path.exists(output_path):
        return []
    print(f"Resuming labels from: {output_path}")
    return torch.load(output_path)


def load_cache(cache_path, resume):
    if not resume or not os.path.exists(cache_path):
        return {}
    print(f"Resuming cache from: {cache_path}")
    return torch.load(cache_path)


def save_outputs(records, output_path, cache, cache_path):
    torch.save(records, output_path)
    torch.save(cache, cache_path)
    print(f"Checkpoint saved | labels: {output_path} | cache: {cache_path} | rows: {len(records)}")


def group_chunks_by_question(dataset, start_question, num_questions):
    grouped = defaultdict(list)
    for item in dataset:
        question_id = int(item["question_id"])
        if question_id < start_question:
            continue
        grouped[question_id].append(item)

    ordered_question_ids = sorted(grouped.keys())
    if num_questions is not None:
        ordered_question_ids = ordered_question_ids[:num_questions]

    grouped_records = []
    for question_id in ordered_question_ids:
        chunks = sorted(grouped[question_id], key=lambda row: int(row["chunk_id"]))
        if chunks and chunks[0].get("is_final_correct") and args_only_small_wrong:
            continue
        grouped_records.append((question_id, chunks))
    return grouped_records


args_only_small_wrong = False


def build_processed_pairs(existing_records):
    processed = set()
    for row in existing_records:
        processed.add((int(row["question_id"]), int(row["chunk_id"])))
    return processed


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


def build_feature_arrays(question_records, feature_spec):
    rows = []
    labels = []
    groups = []
    chunk_refs = []
    for question_id, chunks in question_records:
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            prev_chunk = None if index == 0 else chunks[index - 1]
            rows.append(build_feature_vector(chunk, prev_chunk, total_chunks, feature_spec))
            labels.append(int(chunk["label"]))
            groups.append(question_id)
            chunk_refs.append((question_id, int(chunk["chunk_id"])))
    return np.stack(rows), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64), chunk_refs


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


def score_candidates_with_mlp(question_records, args):
    X, y, groups, chunk_refs = build_feature_arrays(question_records, args.feature_key)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
    train_indices, _ = next(splitter.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_indices])
    X_all = scaler.transform(X)
    y_train = y[train_indices]

    probe = MLPClassifier(
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
    X_fit, y_fit = upsample_minority_class(X_train, y_train, args.random_state)
    probe.fit(X_fit, y_fit)
    prefix_correct_scores = probe.predict_proba(X_all)[:, 1]
    error_scores = 1.0 - prefix_correct_scores

    question_to_chunk_scores = defaultdict(dict)
    for (question_id, chunk_id), score in zip(chunk_refs, error_scores):
        question_to_chunk_scores[int(question_id)][int(chunk_id)] = float(score)
    return question_to_chunk_scores


def load_candidate_scores(question_records, args):
    if args.probe_artifact_path and os.path.exists(args.probe_artifact_path):
        print(f"Loading fixed probe artifact for candidate scoring from: {args.probe_artifact_path}")
        artifact = torch.load(args.probe_artifact_path)
        artifact_feature_key = artifact.get("feature_key", args.feature_key)
        if artifact_feature_key != args.feature_key:
            raise ValueError(
                f"Probe artifact feature_key={artifact_feature_key} does not match requested feature_key={args.feature_key}."
            )
        probe = artifact["probe"]
        scaler = artifact["scaler"]
        X, _, _, chunk_refs = build_feature_arrays(question_records, args.feature_key)
        X_scaled = scaler.transform(X)
        prefix_correct_scores = probe.predict_proba(X_scaled)[:, 1]
        error_scores = 1.0 - prefix_correct_scores
        question_to_chunk_scores = defaultdict(dict)
        for (question_id, chunk_id), score in zip(chunk_refs, error_scores):
            question_to_chunk_scores[int(question_id)][int(chunk_id)] = float(score)
        return question_to_chunk_scores

    print("Probe artifact not found; fitting temporary MLP for candidate scoring.")
    return score_candidates_with_mlp(question_records, args)


def add_explore_candidate(candidate_ids, chunks, position_name):
    if not chunks:
        return
    if position_name == "first":
        candidate_ids.add(int(chunks[0]["chunk_id"]))
    elif position_name == "middle":
        candidate_ids.add(int(chunks[len(chunks) // 2]["chunk_id"]))
    elif position_name == "last":
        candidate_ids.add(int(chunks[-1]["chunk_id"]))
    else:
        raise ValueError(f"Unsupported explore position: {position_name}")


def build_question_candidates(question_records, question_to_chunk_scores, args):
    candidate_map = {}
    explore_positions = parse_csv_tokens(args.explore_positions)

    for question_id, chunks in question_records:
        if args.candidate_mode == "all":
            candidate_map[question_id] = {int(chunk["chunk_id"]) for chunk in chunks}
            continue

        chunk_scores = question_to_chunk_scores.get(question_id, {})
        ranked_ids = sorted(
            (int(chunk["chunk_id"]) for chunk in chunks),
            key=lambda chunk_id: chunk_scores.get(chunk_id, 0.0),
            reverse=True,
        )
        candidate_ids = set(ranked_ids[: max(args.top_k, 0)])

        if args.candidate_mode == "hybrid":
            for position_name in explore_positions:
                add_explore_candidate(candidate_ids, chunks, position_name)

        candidate_map[question_id] = candidate_ids
    return candidate_map


def main():
    args = parse_args()
    global args_only_small_wrong
    args_only_small_wrong = args.only_small_wrong

    print(f"Loading strict labeled chunk data from: {args.input_path}")
    dataset = torch.load(args.input_path)
    question_records = group_chunks_by_question(dataset, args.start_question, args.num_questions)
    print(f"Questions selected for takeover_beneficial labeling: {len(question_records)}")

    question_to_chunk_scores = None
    if args.candidate_mode != "all":
        question_to_chunk_scores = load_candidate_scores(question_records, args)
    candidate_map = build_question_candidates(question_records, question_to_chunk_scores, args)
    candidate_total = sum(len(candidate_ids) for candidate_ids in candidate_map.values())
    total_chunks = sum(len(chunks) for _, chunks in question_records)
    print(f"Candidate chunks selected: {candidate_total}/{total_chunks}")

    print(f"Loading large model from: {args.model_path}")
    large_tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    large_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    large_model.eval()

    print(f"Loading small model from: {args.small_model_path}")
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    output_records = load_output(args.output_path, args.resume)
    processed_pairs = build_processed_pairs(output_records)
    cache = load_cache(args.cache_path, args.resume)

    newly_processed = 0
    for question_id, chunks in tqdm(question_records, desc="Building takeover_beneficial labels"):
        question = chunks[0]["question"]
        ground_truth_final_answer = chunks[0]["ground_truth_final_answer"]
        small_is_correct = bool(chunks[0]["is_final_correct"])
        small_final_answer = chunks[0]["model_final_answer"]
        candidate_ids = candidate_map.get(question_id, set())

        for index, chunk in enumerate(chunks):
            chunk_id = int(chunk["chunk_id"])
            if chunk_id not in candidate_ids:
                continue

            pair_key = (question_id, chunk_id)
            if pair_key in processed_pairs:
                continue

            if index > 0:
                safe_chunk = chunks[index - 1]
                takeover_start_chunk_id = int(safe_chunk["chunk_id"])
                safe_prefix_text = safe_chunk["prefix_text"]
            else:
                takeover_start_chunk_id = -1
                safe_prefix_text = None

            cache_key = (question_id, takeover_start_chunk_id, args.large_handoff_chunks, args.max_new_tokens)
            if cache_key not in cache:
                cache[cache_key] = simulate_local_handoff(
                    question=question,
                    assistant_prefix=safe_prefix_text,
                    small_model=small_model,
                    small_tokenizer=small_tokenizer,
                    large_model=large_model,
                    large_tokenizer=large_tokenizer,
                    args=args,
                )

            takeover_result = cache[cache_key]
            takeover_is_correct = takeover_result["final_answer"] == ground_truth_final_answer
            beneficial = int((not small_is_correct) and takeover_is_correct)
            harmful = int(small_is_correct and (not takeover_is_correct))
            if beneficial:
                takeover_label = "beneficial"
            elif harmful:
                takeover_label = "harmful"
            else:
                takeover_label = "neutral"

            output_records.append(
                {
                    **chunk,
                    "takeover_anchor_chunk_id": takeover_start_chunk_id,
                    "takeover_safe_prefix_text": safe_prefix_text,
                    "takeover_final_answer": takeover_result["final_answer"],
                    "takeover_is_correct": takeover_is_correct,
                    "takeover_generated_token_count": takeover_result["generated_token_count"],
                    "takeover_large_generated_tokens": takeover_result["large_generated_tokens"],
                    "takeover_large_generated_chunks": takeover_result["large_generated_chunks"],
                    "takeover_full_reasoning": takeover_result["full_reasoning"],
                    "small_is_correct": small_is_correct,
                    "small_final_answer": small_final_answer,
                    "candidate_mode": args.candidate_mode,
                    "candidate_error_score": None if question_to_chunk_scores is None else question_to_chunk_scores.get(question_id, {}).get(chunk_id),
                    "takeover_beneficial": beneficial,
                    "takeover_harmful": harmful,
                    "takeover_label": takeover_label,
                }
            )
            processed_pairs.add(pair_key)
            newly_processed += 1

            if args.save_every > 0 and newly_processed % args.save_every == 0:
                save_outputs(output_records, args.output_path, cache, args.cache_path)

    save_outputs(output_records, args.output_path, cache, args.cache_path)


if __name__ == "__main__":
    main()

