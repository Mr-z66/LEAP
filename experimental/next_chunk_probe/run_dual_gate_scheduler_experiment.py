import argparse
import json
import os
import sys
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.config import MODELS, SCHEDULER
from core_package.schedulers.simulate_observe_rollback_scheduler import (
    TorchMLPProbe,
    apply_small_baseline_overrides,
    build_question_records,
    can_apply_large_handoff,
    compute_chunk_risk_score,
    decide_small_chunk_trigger,
    load_probe_artifact,
    parse_csv_floats,
    print_route_trace,
    resolve_system_prompt,
    run_adaptive_large_handoff,
    run_chunk,
    run_large_handoff,
    safe_mean,
    to_jsonable,
)


DEFAULT_BASE_LABEL_PATH = SCHEDULER.label_path
DEFAULT_BASE_PROBE_ARTIFACT_PATH = SCHEDULER.probe_artifact_path
DEFAULT_AUX_LABEL_PATH = "experimental/next_chunk_probe/math500_next2_risk.pt"
DEFAULT_AUX_PROBE_ARTIFACT_PATH = "result/artifacts/math500_next2_risk_probe.pt"
DEFAULT_EVAL_DATA_PATH = "dataset/math500_test_15b_hidden_states_hf_t2048.pt"
DEFAULT_SMALL_MODEL_PATH = MODELS.small_model_path
DEFAULT_LARGE_MODEL_PATH = MODELS.large_model_path
DEFAULT_BASE_THRESHOLDS = "0.50,0.55"
DEFAULT_AUX_THRESHOLDS = "0.25,0.30"
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_MIN_CHUNK_TOKENS = 5
DEFAULT_MAX_CHUNK_TOKENS = 30
DEFAULT_MAX_HANDOFFS = 2
DEFAULT_LARGE_HANDOFF_CHUNKS = 2
DEFAULT_ANSWER_TYPE = "math500_qwen_boxed"
DEFAULT_SYSTEM_PROMPT = MODELS.system_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a dual-gate observe-and-rollback experiment: old probe AND next-2 probe."
    )
    parser.add_argument("--base-label-path", default=DEFAULT_BASE_LABEL_PATH, help="Training label dataset used by the old/base probe artifact.")
    parser.add_argument("--base-probe-artifact-path", default=DEFAULT_BASE_PROBE_ARTIFACT_PATH, help="Old/base probe artifact path.")
    parser.add_argument("--aux-label-path", default=DEFAULT_AUX_LABEL_PATH, help="Training label dataset used by the auxiliary next-2 probe artifact.")
    parser.add_argument("--aux-probe-artifact-path", default=DEFAULT_AUX_PROBE_ARTIFACT_PATH, help="Auxiliary next-2 probe artifact path.")
    parser.add_argument("--eval-data-path", default=DEFAULT_EVAL_DATA_PATH, help="Raw trajectory .pt used for evaluation.")
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--base-thresholds", default=DEFAULT_BASE_THRESHOLDS, help="Comma-separated thresholds for the old/base probe.")
    parser.add_argument("--aux-thresholds", default=DEFAULT_AUX_THRESHOLDS, help="Comma-separated thresholds for the auxiliary next-2 probe.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS)
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS)
    parser.add_argument("--tail-bonus-weight", type=float, default=0.0)
    parser.add_argument("--max-handoffs", type=int, default=DEFAULT_MAX_HANDOFFS)
    parser.add_argument("--large-handoff-chunks", type=int, default=DEFAULT_LARGE_HANDOFF_CHUNKS)
    parser.add_argument("--cooldown-chunks", type=int, default=2)
    parser.add_argument("--require-consecutive-risk", action="store_true")
    parser.add_argument("--num-test-questions", type=int, default=100)
    parser.add_argument("--trace-question-id", type=int, default=None)
    parser.add_argument("--trace-export-path", default="result/traces/observe_rollback_traces_math500_dual_gate.json")
    parser.add_argument("--small-baseline-path", default=None)
    parser.add_argument("--answer-type", default=DEFAULT_ANSWER_TYPE)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--large-backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model-name", default=None)
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument("--adaptive-large-handoff", action="store_true")
    parser.add_argument("--min-large-handoff-chunks", type=int, default=1)
    parser.add_argument("--max-adaptive-large-handoff-chunks", type=int, default=4)
    parser.add_argument("--handoff-recovery-threshold", type=float, default=None)
    return parser.parse_args()


def print_dual_summary(summary):
    print("\nDual-gate observe-and-rollback scheduler simulation")
    print("=" * 50)
    print(f"Base threshold: {summary['base_threshold']:.2f}")
    print(f"Aux threshold: {summary['aux_threshold']:.2f}")
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


def simulate_question_dual(
    record,
    small_model,
    small_tokenizer,
    large_model,
    large_tokenizer,
    base_probe,
    base_scaler,
    base_threshold,
    base_artifact,
    aux_probe,
    aux_scaler,
    aux_threshold,
    aux_artifact,
    args,
):
    question = record["question"]
    ground_truth_final_answer = record["ground_truth_final_answer"]
    answer_extractor = __import__("core_package.answer_registry", fromlist=["get_answer_extractor"]).get_answer_extractor(args.answer_type)
    check_answer_correctness = __import__("core_package.answer_registry", fromlist=["check_answer_correctness"]).check_answer_correctness
    prompt_token_count = __import__("core_package.schedulers.simulate_observe_rollback_scheduler", fromlist=["prompt_token_count"]).prompt_token_count

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
    aux_scores = []
    trigger_progresses = []
    runtime_small_chunks = []
    reset_prev_chunk = False
    cooldown_remaining = 0
    route_trace = []
    previous_base_combined_score = None

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
        runtime_small_chunks.append(small_chunk)

        if reset_prev_chunk:
            prev_chunk = None
            reset_prev_chunk = False
        else:
            prev_chunk = runtime_small_chunks[-2] if len(runtime_small_chunks) >= 2 else None
        total_chunks_for_features = max(len(runtime_small_chunks), int(small_chunk["chunk_id"]) + 1)

        base_risk_info = compute_chunk_risk_score(
            small_chunk,
            prev_chunk=prev_chunk,
            total_chunks_for_features=total_chunks_for_features,
            progress_ratio=progress_ratio,
            probe=base_probe,
            scaler=base_scaler,
            args=args,
            artifact=base_artifact,
        )
        aux_risk_info = compute_chunk_risk_score(
            small_chunk,
            prev_chunk=prev_chunk,
            total_chunks_for_features=total_chunks_for_features,
            progress_ratio=progress_ratio,
            probe=aux_probe,
            scaler=aux_scaler,
            args=args,
            artifact=aux_artifact,
        )
        base_trigger_score = base_risk_info["trigger_score"]
        base_combined_score = base_risk_info["combined_score"]
        aux_trigger_score = aux_risk_info["trigger_score"]
        aux_combined_score = aux_risk_info["combined_score"]
        prev_score_for_trace = previous_base_combined_score

        base_trigger_decision = decide_small_chunk_trigger(
            combined_score=base_combined_score,
            previous_combined_score=previous_base_combined_score,
            threshold=base_threshold,
            args=args,
        )
        meets_base_risk_trigger = base_trigger_decision["meets_risk_trigger"]
        trigger_rule = base_trigger_decision["trigger_rule"]
        meets_aux_risk_trigger = aux_combined_score >= aux_threshold
        meets_risk_trigger = meets_base_risk_trigger and meets_aux_risk_trigger

        if can_apply_large_handoff(meets_risk_trigger, handoff_count, cooldown_remaining, args):
            triggered = True
            trigger_scores.append(base_combined_score)
            aux_scores.append(aux_combined_score)
            trigger_progresses.append(progress_ratio)

            route_trace.append(
                {
                    "event": "small_observe_rollback_dual_gate",
                    "chunk_id": int(small_chunk["chunk_id"]),
                    "chunk_text": small_chunk["chunk_text"],
                    "generated_token_count": small_chunk["generated_token_count"],
                    "base_trigger_score": float(base_trigger_score),
                    "base_combined_score": float(base_combined_score),
                    "aux_trigger_score": float(aux_trigger_score),
                    "aux_combined_score": float(aux_combined_score),
                    "previous_base_combined_score": None if prev_score_for_trace is None else float(prev_score_for_trace),
                    "progress_ratio": float(progress_ratio),
                    "cut_reason": small_chunk["cut_reason"],
                    "trigger_rule": trigger_rule,
                }
            )

            total_tokens -= small_chunk["generated_token_count"]
            runtime_small_chunks.pop()

            if args.adaptive_large_handoff:
                large_result = run_adaptive_large_handoff(
                    question=question,
                    assistant_prefix=safe_prefix,
                    current_total_tokens=total_tokens,
                    next_chunk_index=chunk_index,
                    runtime_small_chunks=runtime_small_chunks,
                    small_model=small_model,
                    small_tokenizer=small_tokenizer,
                    large_model=large_model,
                    large_tokenizer=large_tokenizer,
                    probe=base_probe,
                    scaler=base_scaler,
                    threshold=base_threshold,
                    args=args,
                    artifact=base_artifact,
                )
            else:
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
            if args.adaptive_large_handoff:
                route_trace.append(
                    {
                        "event": "large_handoff",
                        "handoff_index": handoff_count,
                        "mode": "adaptive",
                        "generated_token_count": large_result["generated_token_count"],
                        "generated_chunks": large_result["generated_chunks"],
                    }
                )
                route_trace.extend(large_result["trace_events"])
            else:
                route_trace.append(
                    {
                        "event": "large_handoff",
                        "handoff_index": handoff_count,
                        "mode": "fixed",
                        "generated_token_count": large_result["generated_token_count"],
                        "generated_chunks": large_result["generated_chunks"],
                        "chunks": large_result["chunks"],
                    }
                )
            chunk_index += large_result["generated_chunks"]
            reset_prev_chunk = True
            cooldown_remaining = args.cooldown_chunks
            previous_base_combined_score = None
            if large_result["reached_eos"]:
                break
            continue

        route_trace.append(
            {
                "event": "small_accept_dual_gate",
                "chunk_id": int(small_chunk["chunk_id"]),
                "chunk_text": small_chunk["chunk_text"],
                "generated_token_count": small_chunk["generated_token_count"],
                "base_trigger_score": float(base_trigger_score),
                "base_combined_score": float(base_combined_score),
                "aux_trigger_score": float(aux_trigger_score),
                "aux_combined_score": float(aux_combined_score),
                "previous_base_combined_score": None if prev_score_for_trace is None else float(prev_score_for_trace),
                "progress_ratio": float(progress_ratio),
                "cut_reason": small_chunk["cut_reason"],
                "meets_base_risk_trigger": bool(meets_base_risk_trigger),
                "meets_aux_risk_trigger": bool(meets_aux_risk_trigger),
            }
        )
        prefix = small_chunk["full_reasoning"]
        chunk_index += 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        previous_base_combined_score = base_combined_score
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
        "avg_aux_score": safe_mean(aux_scores),
        "avg_trigger_progress": safe_mean(trigger_progresses),
        "route_trace": route_trace,
    }


def simulate_threshold_dual(
    test_records,
    small_model,
    small_tokenizer,
    large_model,
    large_tokenizer,
    base_probe,
    base_scaler,
    base_threshold,
    base_artifact,
    aux_probe,
    aux_scaler,
    aux_threshold,
    aux_artifact,
    args,
):
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

    from tqdm import tqdm

    progress = tqdm(test_records, desc=f"Base {base_threshold:.2f} Aux {aux_threshold:.2f}", leave=False)
    for record in progress:
        small_is_correct = record["small_is_correct"]
        if small_is_correct:
            small_only_correct += 1
        else:
            error_questions += 1

        result = simulate_question_dual(
            record=record,
            small_model=small_model,
            small_tokenizer=small_tokenizer,
            large_model=large_model,
            large_tokenizer=large_tokenizer,
            base_probe=base_probe,
            base_scaler=base_scaler,
            base_threshold=base_threshold,
            base_artifact=base_artifact,
            aux_probe=aux_probe,
            aux_scaler=aux_scaler,
            aux_threshold=aux_threshold,
            aux_artifact=aux_artifact,
            args=args,
        )

        scheduled_correct += int(result["scheduled_is_correct"])
        if result["triggered"]:
            triggered_questions += 1
            handoff_counts.append(result["handoff_count"])
            large_takeover_tokens.append(result["large_generated_tokens"])
            if not isinstance(result["avg_trigger_progress"], float) or not __import__("math").isnan(result["avg_trigger_progress"]):
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
                "avg_aux_score": result["avg_aux_score"],
                "avg_trigger_progress": result["avg_trigger_progress"],
                "small_final_answer": record["small_final_answer"],
                "scheduled_final_answer": result["scheduled_final_answer"],
                "route_trace": result["route_trace"],
            }
        )

    total_questions = len(test_records)
    correct_questions = total_questions - error_questions
    return {
        "base_threshold": base_threshold,
        "aux_threshold": aux_threshold,
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


def main():
    setattr(sys.modules["__main__"], "TorchMLPProbe", TorchMLPProbe)

    args = parse_args()
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)
    base_thresholds = parse_csv_floats(args.base_thresholds)
    aux_thresholds = parse_csv_floats(args.aux_thresholds)

    print(f"Loading base probe training label dataset from: {args.base_label_path}")
    base_train_dataset = torch.load(args.base_label_path, weights_only=False)
    print(f"Loading fixed base probe artifact from: {args.base_probe_artifact_path}")
    args.probe_artifact_path = args.base_probe_artifact_path
    base_artifact = load_probe_artifact(args)
    if base_artifact is None:
        raise FileNotFoundError(f"Base probe artifact not found: {args.base_probe_artifact_path}")
    if "feature_key" not in base_artifact or not base_artifact.get("feature_key"):
        base_artifact["feature_key"] = "boundary+mean"

    print(f"Loading auxiliary next-2 training label dataset from: {args.aux_label_path}")
    aux_train_dataset = torch.load(args.aux_label_path, weights_only=False)
    print(f"Loading fixed auxiliary next-2 probe artifact from: {args.aux_probe_artifact_path}")
    args.probe_artifact_path = args.aux_probe_artifact_path
    aux_artifact = load_probe_artifact(args)
    if aux_artifact is None:
        raise FileNotFoundError(f"Auxiliary probe artifact not found: {args.aux_probe_artifact_path}")
    if "feature_key" not in aux_artifact or not aux_artifact.get("feature_key"):
        aux_artifact["feature_key"] = "boundary+mean+delta_prev+relative_position"

    base_feature_key = base_artifact.get("feature_key")
    aux_feature_key = aux_artifact.get("feature_key")
    # The legacy scheduler helper eagerly evaluates args.feature_key even when
    # the artifact already carries its own feature_key. Populate a safe default
    # here so old artifacts can still be scored without touching the baseline
    # scheduler implementation.
    args.feature_key = base_feature_key
    args.base_feature_key = base_feature_key
    args.aux_feature_key = aux_feature_key
    print(f"Base feature key: {base_feature_key}")
    print(f"Aux feature key: {aux_feature_key}")

    base_train_question_records = build_question_records(base_train_dataset, base_feature_key)
    apply_small_baseline_overrides(base_train_question_records, args.small_baseline_path)
    aux_train_question_records = build_question_records(aux_train_dataset, aux_feature_key)
    apply_small_baseline_overrides(aux_train_question_records, args.small_baseline_path)

    print(f"Loading separate evaluation dataset from: {args.eval_data_path}")
    eval_dataset = torch.load(args.eval_data_path, weights_only=False)
    eval_question_records = build_question_records(eval_dataset, base_feature_key)
    updated_eval_small_records = apply_small_baseline_overrides(eval_question_records, args.small_baseline_path)
    print(f"Loaded eval questions with base feature key {base_feature_key}: {len(eval_question_records)}")
    if updated_eval_small_records:
        print(f"Overrode stored small baseline for {updated_eval_small_records} eval questions.")

    base_scaler = base_artifact["scaler"]
    base_probe = base_artifact["probe"]
    aux_scaler = aux_artifact["scaler"]
    aux_probe = aux_artifact["probe"]

    base_test_question_ids = [int(question_id) for question_id in base_artifact["test_question_ids"]]
    test_question_ids = [question_id for question_id in base_test_question_ids if question_id in eval_question_records]
    print(
        "Restricting eval to held-out base artifact test split: "
        f"{len(test_question_ids)}/{len(base_test_question_ids)} questions matched."
    )
    test_records = [eval_question_records[question_id] for question_id in sorted(test_question_ids)]
    if args.num_test_questions is not None:
        test_records = test_records[: args.num_test_questions]
        print(f"Using first {len(test_records)} held-out questions for dual-gate scheduler simulation.")

    print(f"Loading small model from: {args.small_model_path}")
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, local_files_only=True)
    if args.large_backend == "hf":
        print(f"Loading large model from: {args.large_model_path}")
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        large_model.eval()
    else:
        large_model = None
        served_name = args.vllm_model_name or os.path.basename(args.large_model_path.rstrip("/\\"))
        print(f"Using vLLM backend for large handoff: {args.vllm_base_url} | served_model={served_name}")

    print(f"Simulating base thresholds: {base_thresholds}")
    print(f"Simulating aux thresholds: {aux_thresholds}")
    summaries = []
    for base_threshold in base_thresholds:
        for aux_threshold in aux_thresholds:
            summary = simulate_threshold_dual(
                test_records=test_records,
                small_model=small_model,
                small_tokenizer=small_tokenizer,
                large_model=large_model,
                large_tokenizer=large_tokenizer,
                base_probe=base_probe,
                base_scaler=base_scaler,
                base_threshold=base_threshold,
                base_artifact=base_artifact,
                aux_probe=aux_probe,
                aux_scaler=aux_scaler,
                aux_threshold=aux_threshold,
                aux_artifact=aux_artifact,
                args=args,
            )
            summaries.append(summary)
            print_dual_summary(summary)

    if args.trace_export_path:
        export_rows = []
        for summary in summaries:
            export_rows.append(
                {
                    "base_threshold": summary["base_threshold"],
                    "aux_threshold": summary["aux_threshold"],
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
                print(
                    f"\nDetailed trace for base_threshold={summary['base_threshold']:.2f} "
                    f"aux_threshold={summary['aux_threshold']:.2f}"
                )
                print_route_trace(matched)
                break

    best_summary = max(summaries, key=lambda item: item["scheduled_gain_over_small"])
    print("\nRecommended dual-gate threshold pair")
    print("=" * 50)
    print(
        f"base_threshold={best_summary['base_threshold']:.2f} | "
        f"aux_threshold={best_summary['aux_threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['scheduled_gain_over_small']:+.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
