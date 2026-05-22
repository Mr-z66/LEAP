import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core_package.answer_registry import check_answer_correctness, get_answer_extractor, resolve_answer_type  # noqa: E402
from core_package.config import MODELS  # noqa: E402
from core_package.gsm8k_protocol import append_gsm8k_boxed_instruction  # noqa: E402
from core_package.math500_protocol import append_math500_instruction  # noqa: E402
from core_package.svamp_protocol import append_svamp_boxed_instruction  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "baselines" / "rsd"
DEFAULT_GSM8K_PATH = REPO_ROOT / "dataset" / "gsm8k_test_300_from_pt.jsonl"
DEFAULT_SVAMP_PATH = REPO_ROOT / "dataset" / "svamp" / "test.jsonl"
DEFAULT_MATH500_PATH = REPO_ROOT / "dataset" / "math500" / "test.jsonl"
DEFAULT_PRM_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
GET_RESPONSES = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSD on LEAP benchmarks and summarize accuracy, cost, and latency.")
    parser.add_argument("--datasets", default="gsm8k,svamp", help="Comma-separated dataset list. Supported: gsm8k,svamp,math500")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory for summaries.")
    parser.add_argument("--gsm8k-path", default=str(DEFAULT_GSM8K_PATH))
    parser.add_argument("--svamp-path", default=str(DEFAULT_SVAMP_PATH))
    parser.add_argument("--math500-path", default=str(DEFAULT_MATH500_PATH))
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap per dataset.")

    parser.add_argument("--draft-model-path", default=MODELS.small_model_path, help="Tokenizer/model path for the draft model.")
    parser.add_argument("--target-model-path", default=MODELS.large_model_path, help="Tokenizer/model path for the target model.")
    parser.add_argument("--prm-model-path", default=DEFAULT_PRM_MODEL, help="Tokenizer/model path for the PRM.")
    parser.add_argument("--draft-served-model-name", default=None, help="OpenAI/vLLM served model name for draft. Defaults to basename of path.")
    parser.add_argument("--target-served-model-name", default=None, help="OpenAI/vLLM served model name for target. Defaults to basename of path.")
    parser.add_argument("--prm-served-model-name", default=None, help="OpenAI/vLLM served model name for PRM. Defaults to basename of path.")
    parser.add_argument("--draft-base-url", default="http://localhost:12340/v1")
    parser.add_argument("--target-base-url", default="http://localhost:12341/v1")
    parser.add_argument("--prm-base-url", default="http://localhost:12342/v1")
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--answer-type", default=None, help="Optional answer protocol override.")
    parser.add_argument("--system-prompt", default=MODELS.boxed_math_system_prompt)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens-per-call", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--step-word", default="\n\n")
    parser.add_argument("--prm-threshold", type=float, default=0.7)
    parser.add_argument("--draft-params-b", type=float, default=1.5)
    parser.add_argument("--target-params-b", type=float, default=7.0)
    parser.add_argument("--prm-params-b", type=float, default=1.5)
    return parser.parse_args()


def served_name(path, override):
    if override:
        return override
    return str(path).rstrip("/\\").replace("\\", "/").split("/")[-1]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_gsm8k_row(row, idx):
    return {
        "question_id": row.get("question_id", row.get("id", idx)),
        "problem": str(row["question"]).strip(),
        "gold_answer_text": str(row["answer"]).strip(),
    }


def normalize_svamp_row(row, idx):
    question = str(row.get("question_concat") or f"{row.get('Body', '')} {row.get('Question', '')}").strip()
    return {
        "question_id": row.get("ID", row.get("id", idx)),
        "problem": question,
        "gold_answer_text": str(row.get("Answer", row.get("answer", ""))).strip(),
    }


def normalize_math500_row(row, idx):
    return {
        "question_id": row.get("unique_id", row.get("id", idx)),
        "problem": str(row["problem"]).strip(),
        "gold_answer_text": str(row["answer"]).strip(),
    }


def load_rows(dataset_name, args):
    if dataset_name == "gsm8k":
        rows = [normalize_gsm8k_row(row, idx) for idx, row in enumerate(load_jsonl(args.gsm8k_path))]
    elif dataset_name == "svamp":
        rows = [normalize_svamp_row(row, idx) for idx, row in enumerate(load_jsonl(args.svamp_path))]
    elif dataset_name == "math500":
        rows = [normalize_math500_row(row, idx) for idx, row in enumerate(load_jsonl(args.math500_path))]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if args.max_questions is not None:
        rows = rows[: args.max_questions]
    return rows


def format_question(question, answer_type):
    if answer_type == "gsm8k_boxed_numeric":
        return append_gsm8k_boxed_instruction(question)
    if answer_type == "svamp_boxed_numeric":
        return append_svamp_boxed_instruction(question)
    if answer_type == "math500_qwen_boxed":
        return append_math500_instruction(question)
    return question


def build_prompt(tokenizer, question, answer_type, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_question(question, answer_type)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def mean(values):
    return statistics.mean(values) if values else 0.0


def summarize_cost(token_counts, turn_info, reward_count, args):
    draft_tokens, target_tokens, discarded_draft_tokens = [int(value) for value in token_counts]
    target_steps = sum(1 for _, client_id in turn_info if int(client_id) == 2)
    draft_steps = sum(1 for _, client_id in turn_info if int(client_id) == 1)
    reasoning_steps = len(turn_info)
    draft_total = draft_tokens + discarded_draft_tokens
    raw_cost = (
        args.draft_params_b * draft_total
        + args.target_params_b * target_tokens
        + args.prm_params_b * reward_count
    )
    return {
        "draft_tokens": draft_tokens,
        "target_tokens": target_tokens,
        "discarded_draft_tokens": discarded_draft_tokens,
        "prm_score_calls": reward_count,
        "draft_steps": draft_steps,
        "target_steps": target_steps,
        "reasoning_steps": reasoning_steps,
        "target_step_fraction": target_steps / reasoning_steps if reasoning_steps else 0.0,
        "param_weighted_token_cost": raw_cost,
    }


def build_rsd_args(args):
    return SimpleNamespace(
        draft_model_name_or_path=served_name(args.draft_model_path, args.draft_served_model_name),
        target_model_name_or_path=served_name(args.target_model_path, args.target_served_model_name),
        prm_name_or_path=served_name(args.prm_model_path, args.prm_served_model_name),
        temperature=args.temperature,
        top_p=1 if args.temperature == 0 else args.top_p,
        max_tokens_per_call=args.max_tokens_per_call,
        step_word=args.step_word,
        prm_threshold=args.prm_threshold,
        max_steps=args.max_steps,
        patience=args.patience,
    )


def run_dataset(dataset_name, clients, tokenizers, args):
    if GET_RESPONSES is None:
        raise RuntimeError("RSD get_responses is not loaded. This should be initialized in main().")
    from tqdm import tqdm

    rows = load_rows(dataset_name, args)
    answer_type = args.answer_type or resolve_answer_type(dataset_name)
    extractor = get_answer_extractor(answer_type)
    output_dir = Path(args.output_root) / (
        f"{dataset_name}_draft{served_name(args.draft_model_path, args.draft_served_model_name)}"
        f"_target{served_name(args.target_model_path, args.target_served_model_name)}"
        f"_prmthr{str(args.prm_threshold).replace('.', 'p')}_{answer_type}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rsd_args = build_rsd_args(args)
    draft_client, target_client, prm_client = clients
    draft_tokenizer, target_tokenizer, prm_tokenizer = tokenizers

    per_question_rows = []
    latencies = []
    correct = 0
    print(f"\n=== Running RSD on {dataset_name} | questions={len(rows)} | threshold={args.prm_threshold} ===")
    for idx, row in tqdm(list(enumerate(rows)), total=len(rows), desc=dataset_name):
        prompt = build_prompt(draft_tokenizer, row["problem"], answer_type, args.system_prompt)
        start = time.time()
        outputs, token_counts, turn_info, rewards = GET_RESPONSES(
            rsd_args,
            draft_client,
            target_client,
            prm_client,
            draft_tokenizer,
            target_tokenizer,
            prm_tokenizer,
            [prompt],
            [row["problem"]],
        )
        elapsed = time.time() - start
        latencies.append(elapsed)
        raw_output = (outputs[0] or "").strip()
        pred_answer, pred_found = extractor(raw_output)
        gold_answer, gold_found = extractor(row["gold_answer_text"])
        if not gold_found:
            gold_answer = row["gold_answer_text"]
        is_correct = pred_found and check_answer_correctness(pred_answer, gold_answer, answer_type)
        correct += int(is_correct)
        cost_stats = summarize_cost(token_counts[0], turn_info[0], len(rewards[0]), args)
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
                "final_answer_text": raw_output,
                "turn_info": turn_info[0],
                "reward": rewards[0],
                "token_counts": token_counts[0],
                **cost_stats,
            }
        )

    total = len(rows)
    summary = {
        "dataset_name": dataset_name,
        "questions_total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "method": "rsd",
        "answer_type": answer_type,
        "draft_model_path": args.draft_model_path,
        "target_model_path": args.target_model_path,
        "prm_model_path": args.prm_model_path,
        "draft_served_model_name": rsd_args.draft_model_name_or_path,
        "target_served_model_name": rsd_args.target_model_name_or_path,
        "prm_served_model_name": rsd_args.prm_name_or_path,
        "draft_base_url": args.draft_base_url,
        "target_base_url": args.target_base_url,
        "prm_base_url": args.prm_base_url,
        "prm_threshold": args.prm_threshold,
        "max_tokens_per_call": args.max_tokens_per_call,
        "max_steps": args.max_steps,
        "step_word": args.step_word,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p90_s": percentile(latencies, 0.9),
        "sec_per_question_mean": statistics.mean(latencies) if latencies else 0.0,
        "avg_draft_tokens": mean([row["draft_tokens"] for row in per_question_rows]),
        "avg_target_tokens": mean([row["target_tokens"] for row in per_question_rows]),
        "avg_discarded_draft_tokens": mean([row["discarded_draft_tokens"] for row in per_question_rows]),
        "avg_prm_score_calls": mean([row["prm_score_calls"] for row in per_question_rows]),
        "avg_draft_steps": mean([row["draft_steps"] for row in per_question_rows]),
        "avg_target_steps": mean([row["target_steps"] for row in per_question_rows]),
        "avg_reasoning_steps": mean([row["reasoning_steps"] for row in per_question_rows]),
        "avg_target_step_fraction": mean([row["target_step_fraction"] for row in per_question_rows]),
        "avg_param_weighted_token_cost": mean([row["param_weighted_token_cost"] for row in per_question_rows]),
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
    output_root.mkdir(parents=True, exist_ok=True)
    overall_json = output_root / "rsd_benchmark_summary.json"
    overall_csv = output_root / "rsd_benchmark_summary.csv"
    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "dataset_name",
        "questions_total",
        "correct",
        "accuracy",
        "answer_type",
        "draft_served_model_name",
        "target_served_model_name",
        "prm_served_model_name",
        "prm_threshold",
        "max_tokens_per_call",
        "max_steps",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "sec_per_question_mean",
        "avg_draft_tokens",
        "avg_target_tokens",
        "avg_discarded_draft_tokens",
        "avg_prm_score_calls",
        "avg_draft_steps",
        "avg_target_steps",
        "avg_reasoning_steps",
        "avg_target_step_fraction",
        "avg_param_weighted_token_cost",
    ]
    with open(overall_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def main():
    global GET_RESPONSES
    args = parse_args()
    from openai import OpenAI
    from transformers import AutoTokenizer
    from main_online import get_responses

    GET_RESPONSES = get_responses
    clients = (
        OpenAI(api_key=args.api_key, base_url=args.draft_base_url),
        OpenAI(api_key=args.api_key, base_url=args.target_base_url),
        OpenAI(api_key=args.api_key, base_url=args.prm_base_url),
    )
    tokenizers = (
        AutoTokenizer.from_pretrained(args.draft_model_path, trust_remote_code=True),
        AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True),
        AutoTokenizer.from_pretrained(args.prm_model_path, trust_remote_code=True),
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summaries = [run_dataset(dataset_name, clients, tokenizers, args) for dataset_name in datasets]
    write_overall_summary(args.output_root, summaries)
    print(f"\nSaved overall summary to: {Path(args.output_root) / 'rsd_benchmark_summary.json'}")
    print(f"Saved overall CSV to: {Path(args.output_root) / 'rsd_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
