import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.config import MODELS
from core_package.schedulers.simulate_observe_rollback_scheduler import (
    apply_small_baseline_overrides,
    build_question_records,
    load_probe_artifact,
    parse_csv_floats,
    print_route_trace,
    print_summary,
    resolve_system_prompt,
    simulate_threshold,
    to_jsonable,
)


DEFAULT_LABEL_PATH = "experimental/next_chunk_probe/math500_next2_risk.pt"
DEFAULT_EVAL_DATA_PATH = "dataset/math500_test_15b_hidden_states_hf_t2048.pt"
DEFAULT_PROBE_ARTIFACT_PATH = "result/artifacts/math500_next2_risk_probe_tuned.pt"
DEFAULT_SMALL_MODEL_PATH = MODELS.small_model_path
DEFAULT_LARGE_MODEL_PATH = MODELS.large_model_path
DEFAULT_THRESHOLDS = "0.20,0.22,0.25"
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_MIN_CHUNK_TOKENS = 5
DEFAULT_MAX_CHUNK_TOKENS = 30
DEFAULT_MAX_HANDOFFS = 2
DEFAULT_LARGE_HANDOFF_CHUNKS = 2
DEFAULT_ANSWER_TYPE = "math500_qwen_boxed"
DEFAULT_SYSTEM_PROMPT = MODELS.system_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run old LEAP scheduler with an experimental next-2 risk probe artifact without touching legacy defaults."
    )
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to next-2 relabeled training dataset used by the probe artifact.")
    parser.add_argument("--eval-data-path", default=DEFAULT_EVAL_DATA_PATH, help="Raw trajectory .pt used for evaluation.")
    parser.add_argument("--probe-artifact-path", default=DEFAULT_PROBE_ARTIFACT_PATH, help="Trained next-2 probe artifact.")
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated probe thresholds to simulate.")
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
    parser.add_argument("--trace-export-path", default="result/traces/observe_rollback_traces_math500_next2_probe.json")
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


def main():
    args = parse_args()
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)
    thresholds = parse_csv_floats(args.thresholds)

    print(f"Loading next-2 probe training label dataset from: {args.label_path}")
    train_dataset = __import__("torch").load(args.label_path, weights_only=False)
    print(f"Loading fixed next-2 probe artifact from: {args.probe_artifact_path}")
    artifact = load_probe_artifact(args)
    if artifact is None:
        raise FileNotFoundError(f"Probe artifact not found: {args.probe_artifact_path}")

    args.feature_key = artifact.get("feature_key")
    train_question_records = build_question_records(train_dataset, args.feature_key)
    updated_small_records = apply_small_baseline_overrides(train_question_records, args.small_baseline_path)
    print(f"Loaded training questions with {args.feature_key}: {len(train_question_records)}")
    if updated_small_records:
        print(f"Overrode stored small baseline for {updated_small_records} training questions.")

    print(f"Loading separate evaluation dataset from: {args.eval_data_path}")
    eval_dataset = __import__("torch").load(args.eval_data_path, weights_only=False)
    eval_question_records = build_question_records(eval_dataset, args.feature_key)
    updated_eval_small_records = apply_small_baseline_overrides(eval_question_records, args.small_baseline_path)
    print(f"Loaded eval questions with {args.feature_key}: {len(eval_question_records)}")
    if updated_eval_small_records:
        print(f"Overrode stored small baseline for {updated_eval_small_records} eval questions.")

    scaler = artifact["scaler"]
    probe = artifact["probe"]
    artifact_test_question_ids = [int(question_id) for question_id in artifact["test_question_ids"]]
    test_question_ids = [question_id for question_id in artifact_test_question_ids if question_id in eval_question_records]
    print(
        "Restricting eval to held-out next-2 artifact test split: "
        f"{len(test_question_ids)}/{len(artifact_test_question_ids)} questions matched."
    )
    test_records = [eval_question_records[question_id] for question_id in sorted(test_question_ids)]
    if args.num_test_questions is not None:
        test_records = test_records[: args.num_test_questions]
        print(f"Using first {len(test_records)} held-out questions for next-2 scheduler simulation.")

    print(f"Loading small model from: {args.small_model_path}")
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=__import__("torch").bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, local_files_only=True)
    if args.large_backend == "hf":
        print(f"Loading large model from: {args.large_model_path}")
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=__import__("torch").bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        large_model.eval()
    else:
        large_model = None
        served_name = args.vllm_model_name or os.path.basename(args.large_model_path.rstrip("/\\"))
        print(f"Using vLLM backend for large handoff: {args.vllm_base_url} | served_model={served_name}")

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
    print("\nRecommended next-2 observe-and-rollback threshold")
    print("=" * 50)
    print(
        f"threshold={best_summary['threshold']:.2f} | "
        f"scheduled_accuracy={best_summary['scheduled_accuracy']:.4f} | "
        f"gain_over_small={best_summary['scheduled_gain_over_small']:+.4f} | "
        f"trigger_rate={best_summary['trigger_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
