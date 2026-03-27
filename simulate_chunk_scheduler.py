import argparse
import csv
import os
import re
import statistics

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_SMALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B")
DEFAULT_LARGE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 55
DEFAULT_FEATURE_KEY = "boundary_hidden_state"
DEFAULT_PROBE_TYPE = "mlp"
DEFAULT_MLP_HIDDEN_LAYERS = "512,128,32"
DEFAULT_MLP_MAX_ITER = 300
DEFAULT_MLP_ALPHA = 1e-4
DEFAULT_MLP_LEARNING_RATE_INIT = 1e-3
DEFAULT_THRESHOLDS = "0.15,0.20,0.25"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_LARGE_BASELINE_MAX_NEW_TOKENS = 768
DEFAULT_TAIL_BONUS_WEIGHT = 0.0
DEFAULT_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step."
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "chunk_scheduler_cache.pt")
DEFAULT_SAVE_CACHE_EVERY = 5
DEFAULT_CASE_EXPORT_DIR = os.path.join(PROJECT_ROOT, "scheduler_case_exports")
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate a chunk-level small-to-large routing system.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled chunk data.")
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--feature-key", default=DEFAULT_FEATURE_KEY, help="Chunk feature used by the probe.")
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
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated risk thresholds to simulate.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max generation tokens for scheduled takeover continuations.",
    )
    parser.add_argument(
        "--large-baseline-max-new-tokens",
        type=int,
        default=DEFAULT_LARGE_BASELINE_MAX_NEW_TOKENS,
        help="Max generation tokens for the full large-model baseline.",
    )
    parser.add_argument(
        "--tail-bonus-weight",
        type=float,
        default=DEFAULT_TAIL_BONUS_WEIGHT,
        help="Add alpha * relative_chunk_position to the probe error score before thresholding.",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Path to cached large-model outputs.")
    parser.add_argument("--save-cache-every", type=int, default=DEFAULT_SAVE_CACHE_EVERY, help="Save cache every N new generations.")
    parser.add_argument(
        "--case-export-dir",
        default=DEFAULT_CASE_EXPORT_DIR,
        help="Directory for exported per-threshold case breakdown CSV files.",
    )
    parser.add_argument(
        "--skip-large-baseline",
        action="store_true",
        help="Skip the full large-model baseline to save compute.",
    )
    return parser.parse_args()


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


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


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {
            "large_baseline": {},
            "takeover": {},
        }
    cache = torch.load(cache_path)
    cache.setdefault("large_baseline", {})
    cache.setdefault("takeover", {})
    return cache


def save_cache(cache, cache_path):
    torch.save(cache, cache_path)
    print(f"Cache saved to: {cache_path}")


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
        r"(?i)final answer\s*[:：]\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n]+)",
        r"(?i)answer\s*[:：]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            explicit_value = extract_last_number(matches[-1])
            if explicit_value is not None:
                return explicit_value

    return extract_last_number(normalize_numeric_text(text))


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

    # Use prefill-style continuation so the model extends the assistant message
    # instead of starting a fresh turn that can leak role markers like "user".
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        continue_final_message=True,
    ), normalized_prefix


def generate_answer(model, tokenizer, question, assistant_prefix, max_new_tokens):
    inputs, normalized_prefix = build_generation_inputs(tokenizer, question, assistant_prefix)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    full_reasoning = normalized_prefix + generated_text if normalized_prefix else generated_text
    final_answer = extract_final_answer(full_reasoning)
    return {
        "generated_text": generated_text,
        "full_reasoning": full_reasoning,
        "final_answer": final_answer,
        "generated_token_count": int(new_token_ids.shape[0]),
    }


def build_question_records(dataset, feature_key):
    question_records = {}
    for item in dataset:
        if feature_key not in item:
            continue
        question_id = int(item["question_id"])
        record = question_records.setdefault(
            question_id,
            {
                "question_id": question_id,
                "question": item["question"],
                "ground_truth_answer_text": item["ground_truth_answer_text"],
                "ground_truth_final_answer": item["ground_truth_final_answer"],
                "small_final_answer": item["model_final_answer"],
                "small_is_correct": bool(item["is_final_correct"]),
                "chunks": [],
            },
        )
        record["chunks"].append(item)

    for record in question_records.values():
        record["chunks"] = sorted(record["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
        record["first_error_chunk_id"] = None
        for chunk in record["chunks"]:
            if int(chunk["label"]) == 0:
                record["first_error_chunk_id"] = int(chunk["chunk_id"])
                break
        total_tokens = sum(int(chunk["token_count"]) for chunk in record["chunks"])
        record["small_total_generated_tokens"] = total_tokens
    return question_records


def build_feature_arrays(question_records, feature_key):
    rows = []
    labels = []
    groups = []
    chunk_refs = []
    for question_id, record in question_records.items():
        for chunk in record["chunks"]:
            rows.append(tensor_to_numpy(chunk[feature_key]))
            labels.append(int(chunk["label"]))
            groups.append(question_id)
            chunk_refs.append((question_id, int(chunk["chunk_id"])))
    X = np.stack(rows)
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups, dtype=np.int64)
    return X, y, g, chunk_refs


def fit_probe_and_score(question_records, args):
    X, y, groups, chunk_refs = build_feature_arrays(question_records, args.feature_key)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_indices, test_indices = next(splitter.split(X, y, groups))

    X_train = X[train_indices]
    y_train = y[train_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)

    probe = build_probe(args)
    if args.probe_type == "mlp":
        X_fit, y_fit = upsample_minority_class(X_train_scaled, y_train, args.random_state)
        probe.fit(X_fit, y_fit)
    else:
        probe.fit(X_train_scaled, y_train)

    prefix_correct_scores = probe.predict_proba(X_scaled)[:, 1]
    error_scores = 1.0 - prefix_correct_scores

    question_to_chunk_scores = {}
    for (question_id, chunk_id), score in zip(chunk_refs, error_scores):
        question_to_chunk_scores.setdefault(question_id, {})[chunk_id] = float(score)

    train_question_ids = sorted(set(int(qid) for qid in groups_train))
    test_question_ids = sorted(set(int(qid) for qid in groups_test))
    return {
        "probe": probe,
        "scaler": scaler,
        "train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "question_to_chunk_scores": question_to_chunk_scores,
    }


def find_trigger_chunk(record, chunk_scores, threshold):
    chunks = record["chunks"]
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk_id = int(chunk["chunk_id"])
        error_score = float(chunk_scores.get(chunk_id, 0.0))
        relative_position = index / max(total_chunks - 1, 1)
        tail_bonus = record.get("tail_bonus_weight", 0.0) * relative_position
        combined_score = error_score + tail_bonus
        if combined_score >= threshold:
            if index > 0:
                safe_chunk = chunks[index - 1]
                takeover_start_chunk_id = int(safe_chunk["chunk_id"])
                safe_prefix_text = safe_chunk["prefix_text"]
                prefix_token_count = int(safe_chunk["end_token_idx"]) + 1
            else:
                takeover_start_chunk_id = -1
                safe_prefix_text = None
                prefix_token_count = 0
            return {
                "trigger_chunk_id": chunk_id,
                "takeover_start_chunk_id": takeover_start_chunk_id,
                "error_score": error_score,
                "tail_bonus": tail_bonus,
                "combined_score": combined_score,
                "chunk_relative_position": relative_position,
                "safe_prefix_text": safe_prefix_text,
                "prefix_token_count": prefix_token_count,
                "remaining_token_count": sum(
                    int(later_chunk["token_count"])
                    for later_chunk in chunks
                    if int(later_chunk["chunk_id"]) > takeover_start_chunk_id
                ),
            }
    return None


def classify_outcome_row(row):
    if not row["small_is_correct"] and row["scheduled_is_correct"]:
        return "small_wrong_to_scheduled_correct"
    if not row["small_is_correct"] and not row["scheduled_is_correct"]:
        return "small_wrong_to_scheduled_wrong"
    if row["small_is_correct"] and not row["scheduled_is_correct"]:
        return "small_correct_to_scheduled_wrong"
    return "small_correct_to_scheduled_correct"


def export_case_rows(summary, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    threshold_tag = str(summary["threshold"]).replace(".", "p")
    base_name = f"threshold_{threshold_tag}"
    all_rows = summary["per_question_rows"]
    if not all_rows:
        return

    fieldnames = [
        "question_id",
        "outcome_type",
        "small_is_correct",
        "large_baseline_is_correct",
        "scheduled_is_correct",
        "triggered",
        "trigger_chunk_id",
        "takeover_start_chunk_id",
        "first_error_chunk_id",
        "trigger_error_score",
        "trigger_tail_bonus",
        "trigger_combined_score",
        "ground_truth_final_answer",
        "small_final_answer",
        "scheduled_final_answer",
        "large_baseline_final_answer",
        "takeover_generated_token_count",
        "takeover_full_reasoning",
        "large_baseline_full_reasoning",
        "question",
    ]

    categorized_rows = {}
    prepared_rows = []
    for row in all_rows:
        prepared_row = dict(row)
        prepared_row["outcome_type"] = classify_outcome_row(prepared_row)
        prepared_rows.append(prepared_row)
        categorized_rows.setdefault(prepared_row["outcome_type"], []).append(prepared_row)

    def write_rows(rows, filename):
        path = os.path.join(export_dir, filename)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(prepared_rows, f"{base_name}_all.csv")
    for outcome_type, rows in categorized_rows.items():
        write_rows(rows, f"{base_name}_{outcome_type}.csv")


def simulate_threshold(
    threshold,
    test_records,
    chunk_scores,
    large_model,
    large_tokenizer,
    args,
    run_large_baseline,
    takeover_cache,
    large_baseline_cache,
):
    scheduled_correct = 0
    large_only_correct = 0
    small_only_correct = 0
    small_wrong_to_scheduled_correct = 0
    small_wrong_to_scheduled_wrong = 0
    small_correct_to_scheduled_wrong = 0
    small_correct_to_scheduled_correct = 0
    triggered_questions = 0
    triggered_wrong_questions = 0
    false_alarm_correct_questions = 0
    error_questions = 0
    error_questions_covered = 0
    near_error_questions_covered = 0
    rescue_upper_bound = 0
    trigger_chunk_ids = []
    trigger_positions = []
    takeover_start_chunk_ids = []
    prefix_tokens_consumed = []
    remaining_tokens_consumed = []
    large_takeover_tokens = []
    per_question_rows = []
    new_cache_entries = 0

    progress = tqdm(
        test_records,
        desc=f"Threshold {threshold:.2f}",
        leave=False,
    )
    for record in progress:
        record["tail_bonus_weight"] = args.tail_bonus_weight
        question_id = record["question_id"]
        question = record["question"]
        ground_truth_final_answer = record["ground_truth_final_answer"]
        small_is_correct = record["small_is_correct"]
        first_error_chunk_id = record["first_error_chunk_id"]
        if not small_is_correct:
            error_questions += 1
        else:
            small_only_correct += 1

        trigger = find_trigger_chunk(record, chunk_scores.get(question_id, {}), threshold)
        scheduled_is_correct = small_is_correct
        scheduled_final_answer = record["small_final_answer"]
        takeover_result = None

        if run_large_baseline:
            if question_id not in large_baseline_cache:
                large_baseline_cache[question_id] = generate_answer(
                    large_model,
                    large_tokenizer,
                    question=question,
                    assistant_prefix=None,
                    max_new_tokens=args.large_baseline_max_new_tokens,
                )
                new_cache_entries += 1
                if args.save_cache_every > 0 and new_cache_entries % args.save_cache_every == 0:
                    save_cache(
                        {
                            "large_baseline": large_baseline_cache,
                            "takeover": takeover_cache,
                        },
                        args.cache_path,
                    )
            large_baseline_result = large_baseline_cache[question_id]
            large_baseline_is_correct = large_baseline_result["final_answer"] == ground_truth_final_answer
            large_only_correct += int(large_baseline_is_correct)
        else:
            large_baseline_result = None
            large_baseline_is_correct = None

        if trigger is not None:
            triggered_questions += 1
            trigger_chunk_ids.append(trigger["trigger_chunk_id"])
            trigger_positions.append(trigger["trigger_chunk_id"] / len(record["chunks"]))
            takeover_start_chunk_ids.append(trigger["takeover_start_chunk_id"])
            prefix_tokens_consumed.append(trigger["prefix_token_count"])
            remaining_tokens_consumed.append(trigger["remaining_token_count"])
            if not small_is_correct:
                triggered_wrong_questions += 1
            else:
                false_alarm_correct_questions += 1

            cache_key = (question_id, trigger["takeover_start_chunk_id"])
            if cache_key not in takeover_cache:
                takeover_cache[cache_key] = generate_answer(
                    large_model,
                    large_tokenizer,
                    question=question,
                    assistant_prefix=trigger["safe_prefix_text"],
                    max_new_tokens=args.max_new_tokens,
                )
                new_cache_entries += 1
                if args.save_cache_every > 0 and new_cache_entries % args.save_cache_every == 0:
                    save_cache(
                        {
                            "large_baseline": large_baseline_cache,
                            "takeover": takeover_cache,
                        },
                        args.cache_path,
                    )
            takeover_result = takeover_cache[cache_key]
            large_takeover_tokens.append(takeover_result["generated_token_count"])
            scheduled_final_answer = takeover_result["final_answer"]
            scheduled_is_correct = scheduled_final_answer == ground_truth_final_answer

            if first_error_chunk_id is not None:
                if trigger["trigger_chunk_id"] <= first_error_chunk_id:
                    error_questions_covered += 1
                    rescue_upper_bound += 1
                if trigger["trigger_chunk_id"] <= first_error_chunk_id + 1:
                    near_error_questions_covered += 1

        scheduled_correct += int(scheduled_is_correct)
        if not small_is_correct and scheduled_is_correct:
            small_wrong_to_scheduled_correct += 1
        elif not small_is_correct and not scheduled_is_correct:
            small_wrong_to_scheduled_wrong += 1
        elif small_is_correct and not scheduled_is_correct:
            small_correct_to_scheduled_wrong += 1
        else:
            small_correct_to_scheduled_correct += 1
        per_question_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "small_is_correct": small_is_correct,
                "large_baseline_is_correct": large_baseline_is_correct,
                "scheduled_is_correct": scheduled_is_correct,
                "triggered": trigger is not None,
                "trigger_chunk_id": None if trigger is None else trigger["trigger_chunk_id"],
                "takeover_start_chunk_id": None if trigger is None else trigger["takeover_start_chunk_id"],
                "first_error_chunk_id": first_error_chunk_id,
                "trigger_error_score": None if trigger is None else trigger["error_score"],
                "trigger_tail_bonus": None if trigger is None else trigger["tail_bonus"],
                "trigger_combined_score": None if trigger is None else trigger["combined_score"],
                "ground_truth_final_answer": ground_truth_final_answer,
                "scheduled_final_answer": scheduled_final_answer,
                "small_final_answer": record["small_final_answer"],
                "large_baseline_final_answer": None if large_baseline_result is None else large_baseline_result["final_answer"],
                "takeover_generated_token_count": None if takeover_result is None else takeover_result["generated_token_count"],
                "takeover_full_reasoning": None if takeover_result is None else takeover_result["full_reasoning"],
                "large_baseline_full_reasoning": None if large_baseline_result is None else large_baseline_result["full_reasoning"],
            }
        )
        progress.set_postfix(
            triggered=triggered_questions,
            cached_large=len(large_baseline_cache),
            cached_takeover=len(takeover_cache),
        )

    if new_cache_entries > 0:
        save_cache(
            {
                "large_baseline": large_baseline_cache,
                "takeover": takeover_cache,
            },
            args.cache_path,
        )

    total_questions = len(test_records)
    correct_questions = total_questions - error_questions
    return {
        "threshold": threshold,
        "tail_bonus_weight": args.tail_bonus_weight,
        "questions_total": total_questions,
        "small_only_accuracy": small_only_correct / total_questions,
        "large_only_accuracy": None if not run_large_baseline else large_only_correct / total_questions,
        "scheduled_accuracy": scheduled_correct / total_questions,
        "trigger_rate": triggered_questions / total_questions,
        "questions_triggered": triggered_questions,
        "avg_trigger_chunk_id": safe_mean(trigger_chunk_ids),
        "avg_trigger_relative_position": safe_mean(trigger_positions),
        "avg_takeover_start_chunk_id": safe_mean(takeover_start_chunk_ids),
        "avg_prefix_tokens_before_trigger": safe_mean(prefix_tokens_consumed),
        "avg_remaining_tokens_at_trigger": safe_mean(remaining_tokens_consumed),
        "avg_large_takeover_tokens": safe_mean(large_takeover_tokens),
        "error_questions_total": error_questions,
        "error_questions_triggered": triggered_wrong_questions,
        "error_coverage_rate": error_questions_covered / error_questions if error_questions else float("nan"),
        "near_error_coverage_rate": near_error_questions_covered / error_questions if error_questions else float("nan"),
        "rescue_upper_bound_rate": rescue_upper_bound / error_questions if error_questions else float("nan"),
        "false_alarm_correct_question_rate": (
            false_alarm_correct_questions / correct_questions if correct_questions else float("nan")
        ),
        "small_wrong_to_scheduled_correct_count": small_wrong_to_scheduled_correct,
        "small_wrong_to_scheduled_wrong_count": small_wrong_to_scheduled_wrong,
        "small_correct_to_scheduled_wrong_count": small_correct_to_scheduled_wrong,
        "small_correct_to_scheduled_correct_count": small_correct_to_scheduled_correct,
        "small_wrong_to_scheduled_correct_rate": small_wrong_to_scheduled_correct / total_questions,
        "small_wrong_to_scheduled_wrong_rate": small_wrong_to_scheduled_wrong / total_questions,
        "small_correct_to_scheduled_wrong_rate": small_correct_to_scheduled_wrong / total_questions,
        "small_correct_to_scheduled_correct_rate": small_correct_to_scheduled_correct / total_questions,
        "scheduled_gain_over_small": (scheduled_correct - small_only_correct) / total_questions,
        "per_question_rows": per_question_rows,
    }


def print_threshold_summary(summary):
    print("\nChunk scheduler simulation")
    print("=" * 50)
    print(f"Threshold: {summary['threshold']:.2f}")
    print(f"Tail bonus weight: {summary['tail_bonus_weight']:.4f}")
    print(f"Questions total: {summary['questions_total']}")
    print(f"Small-only accuracy: {summary['small_only_accuracy']:.4f}")
    if summary["large_only_accuracy"] is not None:
        print(f"Large-only accuracy: {summary['large_only_accuracy']:.4f}")
    print(f"Scheduled accuracy: {summary['scheduled_accuracy']:.4f}")
    print(f"Scheduled gain over small: {summary['scheduled_gain_over_small']:+.4f}")
    print(f"Trigger rate: {summary['trigger_rate']:.4f} ({summary['questions_triggered']}/{summary['questions_total']})")
    print(f"Avg trigger chunk id: {summary['avg_trigger_chunk_id']:.2f}")
    print(f"Avg trigger relative position: {summary['avg_trigger_relative_position']:.4f}")
    print(f"Avg rollback takeover start chunk id: {summary['avg_takeover_start_chunk_id']:.2f}")
    print(f"Avg prefix tokens before trigger: {summary['avg_prefix_tokens_before_trigger']:.2f}")
    print(f"Avg remaining tokens at trigger: {summary['avg_remaining_tokens_at_trigger']:.2f}")
    print(f"Avg large takeover tokens: {summary['avg_large_takeover_tokens']:.2f}")
    print(f"Error questions total: {summary['error_questions_total']}")
    print(f"Error questions triggered: {summary['error_questions_triggered']}")
    print(f"Error coverage rate: {summary['error_coverage_rate']:.4f}")
    print(f"Near-error coverage rate: {summary['near_error_coverage_rate']:.4f}")
    print(f"Potential rescue upper bound: {summary['rescue_upper_bound_rate']:.4f}")
    print(f"False-alarm correct-question rate: {summary['false_alarm_correct_question_rate']:.4f}")
    print("Outcome breakdown:")
    print(
        "  small wrong -> scheduled correct: "
        f"{summary['small_wrong_to_scheduled_correct_count']}/{summary['questions_total']} "
        f"({summary['small_wrong_to_scheduled_correct_rate']:.4f})"
    )
    print(
        "  small wrong -> scheduled wrong: "
        f"{summary['small_wrong_to_scheduled_wrong_count']}/{summary['questions_total']} "
        f"({summary['small_wrong_to_scheduled_wrong_rate']:.4f})"
    )
    print(
        "  small correct -> scheduled wrong: "
        f"{summary['small_correct_to_scheduled_wrong_count']}/{summary['questions_total']} "
        f"({summary['small_correct_to_scheduled_wrong_rate']:.4f})"
    )
    print(
        "  small correct -> scheduled correct: "
        f"{summary['small_correct_to_scheduled_correct_count']}/{summary['questions_total']} "
        f"({summary['small_correct_to_scheduled_correct_rate']:.4f})"
    )


def main():
    args = parse_args()
    thresholds = parse_csv_floats(args.thresholds)

    print(f"Loading labeled chunk dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)
    question_records = build_question_records(dataset, args.feature_key)
    print(f"Loaded questions with {args.feature_key}: {len(question_records)}")

    probe_bundle = fit_probe_and_score(question_records, args)
    train_question_ids = set(probe_bundle["train_question_ids"])
    test_question_ids = set(probe_bundle["test_question_ids"])
    print(f"Train questions: {len(train_question_ids)} | Test questions: {len(test_question_ids)}")

    test_records = [question_records[question_id] for question_id in sorted(test_question_ids)]
    chunk_scores = probe_bundle["question_to_chunk_scores"]

    print(f"Loading large model from: {args.large_model_path}")
    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, local_files_only=True)
    large_model = AutoModelForCausalLM.from_pretrained(
        args.large_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    large_model.eval()

    print(f"Loading generation cache from: {args.cache_path}")
    cache = load_cache(args.cache_path)
    large_baseline_cache = cache["large_baseline"]
    takeover_cache = cache["takeover"]
    print(
        f"Cached entries | large_baseline: {len(large_baseline_cache)} | "
        f"takeover: {len(takeover_cache)}"
    )

    print(f"Simulating thresholds: {thresholds}")
    simulation_summaries = []
    for threshold in thresholds:
        summary = simulate_threshold(
            threshold=threshold,
            test_records=test_records,
            chunk_scores=chunk_scores,
            large_model=large_model,
            large_tokenizer=large_tokenizer,
            args=args,
            run_large_baseline=not args.skip_large_baseline,
            takeover_cache=takeover_cache,
            large_baseline_cache=large_baseline_cache,
        )
        simulation_summaries.append(summary)
        export_case_rows(summary, args.case_export_dir)
        print_threshold_summary(summary)

    best_summary = max(simulation_summaries, key=lambda item: item["scheduled_gain_over_small"])
    print("\nRecommended scheduler threshold")
    print("=" * 50)
    print(
        f"threshold={best_summary['threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['scheduled_gain_over_small']:+.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
