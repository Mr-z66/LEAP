import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_package.answer_registry import (  # noqa: E402
    check_answer_correctness,
    get_answer_extractor,
    resolve_answer_type,
)
from glimp_router import glimprouter, model_names  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "baselines" / "glimprouter"
DEFAULT_SVAMP_PATH = REPO_ROOT / "dataset" / "svamp" / "test.jsonl"
DEFAULT_MATH500_PATH = REPO_ROOT / "dataset" / "math500" / "test.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Run GlimpRouter on LEAP benchmarks and summarize accuracy, cost, and latency.")
    parser.add_argument(
        "--datasets",
        default="gsm8k,svamp,math500",
        help="Comma-separated dataset list. Supported: gsm8k,svamp,math500",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory for raw runs and summaries.")
    parser.add_argument("--repeat-num", type=int, default=1, help="Repeat count per question.")
    parser.add_argument("--score-method", choices=["first_token_entropy", "zeroshot"], default="first_token_entropy")
    parser.add_argument("--score-threshold", type=float, default=1.0, help="GlimpRouter entropy threshold.")
    parser.add_argument("--model-size", default="32b", help="Large model size alias.")
    parser.add_argument("--small-model-size", default="1.5b", help="Small model size alias.")
    parser.add_argument("--small-backend", choices=["api", "hf"], default="api", help="Backend for small-model scoring and generation.")
    parser.add_argument("--small-model-path", default=None, help="Local/HF path for the small model when --small-backend hf.")
    parser.add_argument("--answer-type", default=None, help="Optional answer protocol override, e.g. svamp_boxed_numeric.")
    parser.add_argument("--gsm8k-token-budget", type=int, default=2048)
    parser.add_argument("--svamp-token-budget", type=int, default=2048)
    parser.add_argument("--math500-token-budget", type=int, default=4096)
    parser.add_argument("--gsm8k-path", default=None, help="Optional local GSM8K json/jsonl path.")
    parser.add_argument("--svamp-path", default=str(DEFAULT_SVAMP_PATH))
    parser.add_argument("--math500-path", default=str(DEFAULT_MATH500_PATH))
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap per dataset.")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_gsm8k_row(row):
    return {
        "problem": str(row["question"]).strip(),
        "gold_answer_text": str(row["answer"]).strip(),
        "question_id": row.get("question_id", row.get("id", "")),
    }


def normalize_svamp_row(row):
    question = str(row.get("question_concat") or f"{row.get('Body', '')} {row.get('Question', '')}").strip()
    return {
        "problem": question,
        "gold_answer_text": str(row.get("Answer", "")).strip(),
        "question_id": row.get("ID", ""),
    }


def normalize_math500_row(row):
    return {
        "problem": str(row["problem"]).strip(),
        "gold_answer_text": str(row["answer"]).strip(),
        "question_id": row.get("unique_id", ""),
    }


def load_rows(dataset_name, args):
    if dataset_name == "gsm8k":
        if args.gsm8k_path:
            rows = load_jsonl(args.gsm8k_path)
        else:
            rows = list(load_dataset("gsm8k", "main", split="test"))
        rows = [normalize_gsm8k_row(row) for row in rows]
    elif dataset_name == "svamp":
        rows = [normalize_svamp_row(row) for row in load_jsonl(args.svamp_path)]
    elif dataset_name == "math500":
        rows = [normalize_math500_row(row) for row in load_jsonl(args.math500_path)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if args.max_questions is not None:
        rows = rows[: args.max_questions]
    return rows


def extract_answer(result):
    return result[-1]["step_str"]


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def model_size_to_params_b(size_text, default):
    text = str(size_text).lower().replace("b", "")
    try:
        return float(text)
    except ValueError:
        return default


def summarize_metadata(metadata, small_params_b=1.5, large_params_b=32.0):
    reasoning_steps = metadata[:-1]
    small_tokens = sum(int(step.get("num_output_tokens_small") or 0) for step in metadata)
    large_tokens = sum(int(step.get("num_output_tokens_base") or 0) for step in metadata)
    score_calls = sum(1 for step in reasoning_steps if step.get("score") is not None)
    large_steps = sum(1 for step in reasoning_steps if step.get("base_model_step") is not None)
    total_reasoning_steps = len(reasoning_steps)
    cost_proxy = small_params_b * (small_tokens + score_calls) + large_params_b * large_tokens
    return {
        "small_tokens": small_tokens,
        "large_tokens": large_tokens,
        "score_calls": score_calls,
        "large_steps": large_steps,
        "reasoning_steps": total_reasoning_steps,
        "large_step_fraction": (large_steps / total_reasoning_steps) if total_reasoning_steps else 0.0,
        "param_weighted_token_cost": cost_proxy,
    }


def dataset_token_budget(dataset_name, args):
    if dataset_name == "gsm8k":
        return args.gsm8k_token_budget
    if dataset_name == "svamp":
        return args.svamp_token_budget
    if dataset_name == "math500":
        return args.math500_token_budget
    raise ValueError(dataset_name)


def run_dataset(dataset_name, args):
    rows = load_rows(dataset_name, args)
    answer_type = args.answer_type or resolve_answer_type(dataset_name)
    extractor = get_answer_extractor(answer_type)
    run_tag = f"{dataset_name}_large{args.model_size}_small{args.small_model_size}_thr{str(args.score_threshold).replace('.', 'p')}"
    if args.small_backend != "api":
        run_tag += f"_small{args.small_backend}"
    if args.answer_type:
        run_tag += f"_{args.answer_type}"
    output_dir = Path(args.output_root) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    per_question_rows = []
    latencies = []
    correct = 0
    total = len(rows)

    print(f"\n=== Running GlimpRouter on {dataset_name} | questions={total} ===")
    for idx, row in tqdm(list(enumerate(rows)), total=total, desc=dataset_name):
        start = time.time()
        metadata = glimprouter(
            problem={"problem": row["problem"], "question_content": row["problem"], "starter_code": ""},
            options="math",
            dataset_name=dataset_name,
            token_budget=dataset_token_budget(dataset_name, args),
            repeat_id=0,
            score_method=args.score_method,
            output_dir=str(output_dir),
            problem_id=idx,
            score_threshold=args.score_threshold,
            first_n_steps_base_model=16384 if args.score_method == "zeroshot" else 0,
            model_size=args.model_size,
            small_model_size=args.small_model_size,
            small_backend=args.small_backend,
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        raw_answer = extract_answer(metadata)
        pred_answer, pred_found = extractor(raw_answer)
        gold_answer, gold_found = extractor(row["gold_answer_text"])
        if not gold_found:
            gold_answer = row["gold_answer_text"]
        is_correct = pred_found and check_answer_correctness(pred_answer, gold_answer, answer_type)
        correct += int(is_correct)

        cost_stats = summarize_metadata(
            metadata,
            small_params_b=model_size_to_params_b(args.small_model_size, 1.5),
            large_params_b=model_size_to_params_b(args.model_size, 32.0),
        )
        per_question_rows.append(
            {
                "index": idx,
                "question_id": row["question_id"],
                "predicted_answer": pred_answer,
                "pred_found": pred_found,
                "gold_answer": gold_answer,
                "gold_found": gold_found,
                "is_correct": is_correct,
                "latency_s": elapsed,
                **cost_stats,
            }
        )

    summary = {
        "dataset_name": dataset_name,
        "questions_total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "repeat_num": args.repeat_num,
        "score_method": args.score_method,
        "score_threshold": args.score_threshold,
        "model_size": args.model_size,
        "small_model_size": args.small_model_size,
        "small_backend": args.small_backend,
        "answer_type": answer_type,
        "token_budget": dataset_token_budget(dataset_name, args),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p90_s": percentile(latencies, 0.9),
        "sec_per_question_mean": (statistics.mean(latencies) if latencies else 0.0),
        "avg_small_tokens": statistics.mean([row["small_tokens"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_large_tokens": statistics.mean([row["large_tokens"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_score_calls": statistics.mean([row["score_calls"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_large_steps": statistics.mean([row["large_steps"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_reasoning_steps": statistics.mean([row["reasoning_steps"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_large_step_fraction": statistics.mean([row["large_step_fraction"] for row in per_question_rows]) if per_question_rows else 0.0,
        "avg_param_weighted_token_cost": statistics.mean([row["param_weighted_token_cost"] for row in per_question_rows]) if per_question_rows else 0.0,
        "rows": per_question_rows,
    }

    summary_path = output_dir / f"{dataset_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[{dataset_name}] accuracy={summary['accuracy']:.4f} | "
        f"latency_mean_s={summary['latency_mean_s']:.4f} | "
        f"avg_cost={summary['avg_param_weighted_token_cost']:.2f}"
    )
    return summary


def write_overall_summary(output_root, summaries):
    output_root = Path(output_root)
    overall_json = output_root / "glimprouter_benchmark_summary.json"
    overall_csv = output_root / "glimprouter_benchmark_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "dataset_name",
        "questions_total",
        "correct",
        "accuracy",
        "model_size",
        "small_model_size",
        "small_backend",
        "answer_type",
        "score_method",
        "score_threshold",
        "token_budget",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "sec_per_question_mean",
        "avg_small_tokens",
        "avg_large_tokens",
        "avg_score_calls",
        "avg_large_steps",
        "avg_reasoning_steps",
        "avg_large_step_fraction",
        "avg_param_weighted_token_cost",
    ]
    with open(overall_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary.get(k) for k in fieldnames})


def main():
    args = parse_args()
    if args.small_model_path:
        model_names[args.small_model_size] = args.small_model_path
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summaries = [run_dataset(dataset_name, args) for dataset_name in datasets]
    write_overall_summary(args.output_root, summaries)
    print(f"\nSaved overall summary to: {Path(args.output_root) / 'glimprouter_benchmark_summary.json'}")
    print(f"Saved overall CSV to: {Path(args.output_root) / 'glimprouter_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
