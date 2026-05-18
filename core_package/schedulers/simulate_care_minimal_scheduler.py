import argparse
import json
from collections import deque

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.answer_registry import check_answer_correctness, get_answer_extractor
from core_package.config import MODELS, SCHEDULER
from core_package.schedulers.simulate_observe_rollback_scheduler import (
    DEFAULT_ANSWER_TYPE,
    DEFAULT_LABEL_PATH,
    DEFAULT_LARGE_HANDOFF_CHUNKS,
    DEFAULT_LARGE_MODEL_PATH,
    DEFAULT_MAX_HANDOFFS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_MIN_CHUNK_TOKENS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SMALL_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEST_SIZE,
    apply_small_baseline_overrides,
    build_question_records,
    parse_csv_floats,
    prompt_token_count,
    resolve_system_prompt,
    run_chunk,
    run_large_handoff,
    safe_mean,
    to_jsonable,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a training-free CARE-minimal chunk scheduler using entropy and entropy-spread baselines."
    )
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled chunk data or raw trajectory .pt.")
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Optional separate evaluation dataset path. Can be a strict labeled .pt or a raw trajectory .pt from build_dataset.",
    )
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Held-out question ratio.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for the split.")
    parser.add_argument("--thresholds", default="1.0,1.5,2.0", help="Comma-separated entropy z-score thresholds to simulate.")
    parser.add_argument("--var-thresholds", default="1.0,1.5,2.0", help="Comma-separated entropy-spread z-score thresholds to simulate.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max total generation tokens.")
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS, help="Min tokens before punctuation can end a chunk.")
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS, help="Forced chunk cut length.")
    parser.add_argument("--max-handoffs", type=int, default=DEFAULT_MAX_HANDOFFS, help="Maximum number of large-model interventions.")
    parser.add_argument("--large-handoff-chunks", type=int, default=DEFAULT_LARGE_HANDOFF_CHUNKS, help="How many chunks large model handles per intervention.")
    parser.add_argument("--cooldown-chunks", type=int, default=SCHEDULER.cooldown_chunks, help="How many accepted small-model chunks to wait before another handoff is allowed.")
    parser.add_argument(
        "--require-consecutive-risk",
        action="store_true",
        help="Require two consecutive risky chunks before triggering handoff.",
    )
    parser.add_argument("--baseline-warmup-chunks", type=int, default=3, help="How many accepted small chunks to observe before CARE can trigger.")
    parser.add_argument("--baseline-window", type=int, default=6, help="Rolling window size for context-anchored entropy statistics.")
    parser.add_argument(
        "--care-combine-mode",
        choices=["and", "or"],
        default="and",
        help="Whether both entropy and entropy-spread z-scores must exceed threshold.",
    )
    parser.add_argument("--num-test-questions", type=int, default=None, help="Optional cap on held-out test questions.")
    parser.add_argument("--trace-question-id", type=int, default=None, help="Optional question_id to print a detailed chunk routing trace for.")
    parser.add_argument("--trace-export-path", default=None, help="Optional JSON path to export per-question routing traces.")
    parser.add_argument(
        "--small-baseline-path",
        default=None,
        help="Optional JSON baseline to override stored small-model final answers/correctness.",
    )
    parser.add_argument("--answer-type", default=DEFAULT_ANSWER_TYPE, help="Answer protocol used for extraction and correctness.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt used for generation.")
    parser.add_argument(
        "--large-backend",
        choices=["hf", "vllm"],
        default="hf",
        help="Backend used for large-model handoff generation.",
    )
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000", help="Base URL for an OpenAI-compatible vLLM server.")
    parser.add_argument("--vllm-api-key", default="EMPTY", help="API key for the vLLM OpenAI-compatible server.")
    parser.add_argument("--vllm-model-name", default=None, help="Served model name exposed by the vLLM server.")
    parser.add_argument("--vllm-timeout", type=float, default=300.0, help="HTTP timeout in seconds for vLLM calls.")
    return parser.parse_args()


def load_question_records(args):
    dataset_path = args.eval_data_path or args.label_path
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    question_records = build_question_records(dataset, "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob")
    updated = apply_small_baseline_overrides(question_records, args.small_baseline_path)
    if updated:
        print(f"Applied {updated} small-model baseline overrides from {args.small_baseline_path}")
    return question_records


def split_test_records(question_records, args):
    question_ids = sorted(question_records.keys())
    if not question_ids:
        return []
    if args.test_size <= 0 or args.test_size >= 1:
        selected = question_ids
    else:
        dummy_X = np.zeros((len(question_ids), 1), dtype=np.float32)
        dummy_y = np.zeros(len(question_ids), dtype=np.int64)
        groups = np.asarray(question_ids, dtype=np.int64)
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        _, test_indices = next(splitter.split(dummy_X, dummy_y, groups))
        selected = [question_ids[idx] for idx in test_indices]
    if args.num_test_questions is not None:
        selected = selected[: args.num_test_questions]
    return [question_records[qid] for qid in selected]


def chunk_entropy_stats(chunk):
    def scalar(name):
        value = chunk.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(value)

    mean_entropy = scalar("mean_entropy")
    max_entropy = scalar("max_entropy")
    final_entropy = scalar("final_entropy")
    entropy_spread = max(0.0, max_entropy - mean_entropy)
    return {
        "mean_entropy": mean_entropy,
        "final_entropy": final_entropy,
        "entropy_spread": entropy_spread,
    }


def robust_zscore(current_value, history_values):
    if len(history_values) < 2:
        return 0.0
    mean_value = float(np.mean(history_values))
    std_value = float(np.std(history_values))
    if std_value < 1e-6:
        return 0.0
    return float((current_value - mean_value) / (std_value + 1e-6))


def compute_care_risk_score(chunk, entropy_history, spread_history):
    stats = chunk_entropy_stats(chunk)
    z_entropy = robust_zscore(stats["mean_entropy"], list(entropy_history))
    z_spread = robust_zscore(stats["entropy_spread"], list(spread_history))
    return {
        "mean_entropy": stats["mean_entropy"],
        "final_entropy": stats["final_entropy"],
        "entropy_spread": stats["entropy_spread"],
        "z_entropy": z_entropy,
        "z_spread": z_spread,
    }


def care_trigger_decision(risk_info, previous_trigger_state, entropy_threshold, spread_threshold, args):
    entropy_hit = risk_info["z_entropy"] >= entropy_threshold
    spread_hit = risk_info["z_spread"] >= spread_threshold
    if args.care_combine_mode == "and":
        raw_trigger = entropy_hit and spread_hit
        trigger_rule = "entropy_and_spread"
    else:
        raw_trigger = entropy_hit or spread_hit
        trigger_rule = "entropy_or_spread"

    if args.require_consecutive_risk:
        meets_trigger = raw_trigger and previous_trigger_state
        trigger_rule = f"consecutive_{trigger_rule}"
    else:
        meets_trigger = raw_trigger

    combined_score = max(risk_info["z_entropy"] - entropy_threshold, risk_info["z_spread"] - spread_threshold)
    return {
        "meets_risk_trigger": bool(meets_trigger),
        "raw_trigger": bool(raw_trigger),
        "trigger_rule": trigger_rule,
        "combined_score": float(combined_score),
    }


def simulate_question(record, small_model, small_tokenizer, large_model, large_tokenizer, entropy_threshold, spread_threshold, args):
    question = record["question"]
    ground_truth_final_answer = record["ground_truth_final_answer"]
    answer_extractor = get_answer_extractor(args.answer_type)
    question_prompt_token_count = prompt_token_count(
        small_tokenizer,
        question,
        args.system_prompt,
        args.answer_type,
    )

    prefix = None
    total_tokens = 0
    total_large_tokens = 0
    handoff_count = 0
    chunk_index = 0
    triggered = False
    trigger_scores = []
    trigger_progresses = []
    cooldown_remaining = 0
    route_trace = []

    entropy_history = deque(maxlen=args.baseline_window)
    spread_history = deque(maxlen=args.baseline_window)
    previous_raw_trigger = False

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
            answer_type=args.answer_type,
        )

        if small_chunk["generated_token_count"] == 0:
            prefix = safe_prefix
            break

        total_tokens += small_chunk["generated_token_count"]
        progress_ratio = total_tokens / max(args.max_new_tokens, 1)
        small_chunk["chunk_id"] = chunk_index

        risk_info = compute_care_risk_score(small_chunk, entropy_history, spread_history)
        baseline_ready = len(entropy_history) >= args.baseline_warmup_chunks and len(spread_history) >= args.baseline_warmup_chunks
        if baseline_ready:
            trigger_decision = care_trigger_decision(
                risk_info,
                previous_trigger_state=previous_raw_trigger,
                entropy_threshold=entropy_threshold,
                spread_threshold=spread_threshold,
                args=args,
            )
        else:
            trigger_decision = {
                "meets_risk_trigger": False,
                "raw_trigger": False,
                "trigger_rule": "warmup_baseline",
                "combined_score": 0.0,
            }

        can_handoff = (
            trigger_decision["meets_risk_trigger"]
            and handoff_count < args.max_handoffs
            and cooldown_remaining <= 0
        )

        if can_handoff:
            triggered = True
            trigger_scores.append(trigger_decision["combined_score"])
            trigger_progresses.append(progress_ratio)
            route_trace.append(
                {
                    "event": "small_observe_rollback",
                    "chunk_id": int(small_chunk["chunk_id"]),
                    "chunk_text": small_chunk["chunk_text"],
                    "generated_token_count": small_chunk["generated_token_count"],
                    "mean_entropy": risk_info["mean_entropy"],
                    "final_entropy": risk_info["final_entropy"],
                    "entropy_spread": risk_info["entropy_spread"],
                    "z_entropy": risk_info["z_entropy"],
                    "z_spread": risk_info["z_spread"],
                    "combined_score": trigger_decision["combined_score"],
                    "progress_ratio": float(progress_ratio),
                    "cut_reason": small_chunk["cut_reason"],
                    "trigger_rule": trigger_decision["trigger_rule"],
                }
            )

            total_tokens -= small_chunk["generated_token_count"]
            remaining_handoff_budget = max(args.max_new_tokens - total_tokens, 1)
            large_result = run_large_handoff(
                model=large_model,
                tokenizer=large_tokenizer,
                question=question,
                assistant_prefix=safe_prefix,
                args=args,
                num_chunks=args.large_handoff_chunks,
                max_total_new_tokens=remaining_handoff_budget,
            )
            prefix = large_result["full_reasoning"]
            total_tokens += large_result["generated_token_count"]
            total_large_tokens += large_result["generated_token_count"]
            handoff_count += 1
            route_trace.append(
                {
                    "event": "large_handoff",
                    "handoff_index": handoff_count,
                    "mode": "care_minimal",
                    "generated_token_count": large_result["generated_token_count"],
                    "generated_chunks": large_result["generated_chunks"],
                    "chunks": large_result["chunks"],
                }
            )
            chunk_index += large_result["generated_chunks"]
            cooldown_remaining = args.cooldown_chunks
            previous_raw_trigger = False
            if large_result["reached_eos"]:
                break
            continue

        route_trace.append(
            {
                "event": "small_accept",
                "chunk_id": int(small_chunk["chunk_id"]),
                "chunk_text": small_chunk["chunk_text"],
                "generated_token_count": small_chunk["generated_token_count"],
                "mean_entropy": risk_info["mean_entropy"],
                "final_entropy": risk_info["final_entropy"],
                "entropy_spread": risk_info["entropy_spread"],
                "z_entropy": risk_info["z_entropy"],
                "z_spread": risk_info["z_spread"],
                "combined_score": trigger_decision["combined_score"],
                "progress_ratio": float(progress_ratio),
                "cut_reason": small_chunk["cut_reason"],
                "baseline_ready": baseline_ready,
            }
        )
        prefix = small_chunk["full_reasoning"]
        entropy_history.append(risk_info["mean_entropy"])
        spread_history.append(risk_info["entropy_spread"])
        chunk_index += 1
        previous_raw_trigger = trigger_decision["raw_trigger"]
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if small_chunk["reached_eos"]:
            break

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


def simulate_threshold(test_records, small_model, small_tokenizer, large_model, large_tokenizer, entropy_threshold, spread_threshold, args):
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

    progress = tqdm(test_records, desc=f"E{entropy_threshold:.2f}|V{spread_threshold:.2f}", leave=False)
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
            entropy_threshold=entropy_threshold,
            spread_threshold=spread_threshold,
            args=args,
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
                "question_id": int(record["question_id"]),
                "small_is_correct": bool(small_is_correct),
                "scheduled_is_correct": bool(result["scheduled_is_correct"]),
                "triggered": bool(result["triggered"]),
                "handoff_count": int(result["handoff_count"]),
                "avg_trigger_score": result["avg_trigger_score"],
                "avg_trigger_progress": result["avg_trigger_progress"],
                "large_generated_tokens": int(result["large_generated_tokens"]),
                "scheduled_final_answer": result["scheduled_final_answer"],
                "small_final_answer": record.get("small_final_answer"),
                "ground_truth_final_answer": record.get("ground_truth_final_answer"),
                "route_trace": result["route_trace"],
            }
        )

    total_questions = len(test_records)
    return {
        "entropy_threshold": float(entropy_threshold),
        "var_threshold": float(spread_threshold),
        "questions_total": total_questions,
        "small_only_accuracy": small_only_correct / total_questions if total_questions else float("nan"),
        "scheduled_accuracy": scheduled_correct / total_questions if total_questions else float("nan"),
        "gain_over_small": (scheduled_correct - small_only_correct) / total_questions if total_questions else float("nan"),
        "error_questions_total": error_questions,
        "trigger_rate": triggered_questions / total_questions if total_questions else float("nan"),
        "questions_triggered": triggered_questions,
        "error_questions_triggered": triggered_wrong_questions,
        "false_alarm_correct_questions": false_alarm_correct_questions,
        "avg_handoff_count": safe_mean(handoff_counts),
        "avg_large_takeover_tokens": safe_mean(large_takeover_tokens),
        "avg_trigger_progress": safe_mean(trigger_progresses),
        "per_question_rows": per_question_rows,
    }


def print_threshold_summary(summary):
    print(f"Entropy threshold: {summary['entropy_threshold']:.2f}")
    print(f"Var threshold: {summary['var_threshold']:.2f}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Small-only accuracy: {summary['small_only_accuracy']:.4f}")
    print(f"Scheduled accuracy: {summary['scheduled_accuracy']:.4f}")
    print(f"Gain over small: {summary['gain_over_small']:.4f}")
    print(f"Trigger rate: {summary['trigger_rate']:.4f} ({summary['questions_triggered']}/{summary['questions_total']})")
    print(f"Error questions triggered: {summary['error_questions_triggered']}")
    print(f"False alarms on correct questions: {summary['false_alarm_correct_questions']}")
    print(f"Avg handoff count: {summary['avg_handoff_count']:.2f}")
    print(f"Avg trigger progress: {summary['avg_trigger_progress']:.4f}")
    print(f"Avg large takeover tokens: {summary['avg_large_takeover_tokens']:.2f}")


def load_models(args):
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, use_fast=False)
    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, use_fast=False)
    if small_tokenizer.pad_token_id is None:
        small_tokenizer.pad_token = small_tokenizer.eos_token
    if large_tokenizer.pad_token_id is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token

    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.large_backend == "hf":
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        large_model = None
        served_name = args.vllm_model_name or args.large_model_path
        print(f"Using vLLM backend for large handoff: {args.vllm_base_url} | served_model={served_name}")
    return small_model, small_tokenizer, large_model, large_tokenizer


def main():
    args = parse_args()
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)

    question_records = load_question_records(args)
    test_records = split_test_records(question_records, args)
    if not test_records:
        raise RuntimeError("No evaluation questions found.")

    print(f"Loaded question records: {len(question_records)} | test questions: {len(test_records)}")
    entropy_thresholds = parse_csv_floats(args.thresholds)
    var_thresholds = parse_csv_floats(args.var_thresholds)

    small_model, small_tokenizer, large_model, large_tokenizer = load_models(args)

    summaries = []
    best_summary = None
    print(f"Simulating CARE-minimal thresholds: entropy={entropy_thresholds} | var={var_thresholds}")
    for entropy_threshold in entropy_thresholds:
        for spread_threshold in var_thresholds:
            summary = simulate_threshold(
                test_records=test_records,
                small_model=small_model,
                small_tokenizer=small_tokenizer,
                large_model=large_model,
                large_tokenizer=large_tokenizer,
                entropy_threshold=entropy_threshold,
                spread_threshold=spread_threshold,
                args=args,
            )
            summaries.append(summary)
            print_threshold_summary(summary)
            if best_summary is None or summary["scheduled_accuracy"] > best_summary["scheduled_accuracy"]:
                best_summary = summary

    if args.trace_question_id is not None and best_summary is not None:
        matched = [row for row in best_summary["per_question_rows"] if row["question_id"] == args.trace_question_id]
        if matched:
            print(f"\nDetailed CARE trace for question_id={args.trace_question_id}")
            print(json.dumps(to_jsonable(matched[0]), ensure_ascii=False, indent=2))

    if args.trace_export_path:
        export_payload = {
            "mode": "care_minimal_proxy",
            "summaries": to_jsonable(summaries),
        }
        with open(args.trace_export_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved CARE traces to {args.trace_export_path}")

    print("\nRecommended CARE-minimal thresholds")
    print(
        f"entropy_threshold={best_summary['entropy_threshold']:.2f} | "
        f"var_threshold={best_summary['var_threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['gain_over_small']:.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
