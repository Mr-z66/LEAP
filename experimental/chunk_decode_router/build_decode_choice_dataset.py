import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.answer_registry import (
    check_answer_correctness,
    get_answer_extractor,
    resolve_answer_type,
)
from core_package.config import MODELS
from core_package.schedulers.simulate_observe_rollback_scheduler import (
    DEFAULT_ANSWER_TYPE,
    DEFAULT_SYSTEM_PROMPT,
    resolve_system_prompt,
    run_chunk,
    run_large_handoff,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAJECTORY_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "math500_test_15b_hidden_states_hf_t2048.pt"
)
DEFAULT_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "experimental", "chunk_decode_router", "math500_test100_decode_choice_candidates.jsonl"
)
DEFAULT_SMALL_MODEL_PATH = MODELS.small_model_path
DEFAULT_LARGE_MODEL_PATH = MODELS.large_model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build candidate current-chunk decode-choice samples and optionally label them "
            "with SLM-vs-LLM local rollout comparisons."
        )
    )
    parser.add_argument("--trajectory-path", default=DEFAULT_TRAJECTORY_PATH, help="Path to build_dataset .pt trajectories.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output path (.jsonl or .pt).")
    parser.add_argument("--dataset-name", default="math500", choices=["gsm8k", "svamp", "math500", "jsonl"])
    parser.add_argument("--answer-type", default=None, help="Optional answer protocol override.")
    parser.add_argument("--num-questions", type=int, default=100, help="How many questions to keep.")
    parser.add_argument("--question-offset", type=int, default=0, help="Start offset within the trajectory dataset.")
    parser.add_argument("--question-ids", default=None, help="Optional comma-separated question ids to keep.")
    parser.add_argument(
        "--candidate-policy",
        default="uniform_plus_ends",
        choices=["uniform_plus_ends", "risk_guided_plus_ends", "all"],
        help="How to choose candidate current-chunk boundaries per question.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=4,
        help="Target number of internal candidate chunk boundaries per question for the uniform policy.",
    )
    parser.add_argument(
        "--wrong-question-candidate-count",
        type=int,
        default=None,
        help="Optional override for how many internal candidate chunk boundaries to keep when the small model is wrong.",
    )
    parser.add_argument(
        "--correct-question-candidate-count",
        type=int,
        default=None,
        help="Optional override for how many internal candidate chunk boundaries to keep when the small model is correct.",
    )
    parser.add_argument(
        "--label-with-rollouts",
        action="store_true",
        help="Run local SLM/LLM comparisons and emit decode-choice labels.",
    )
    parser.add_argument(
        "--skip-empty-current-chunk",
        action="store_true",
        default=True,
        help=(
            "Skip rollout samples where the re-generated current chunk is empty for either branch. "
            "Enabled by default to avoid end-of-trajectory pseudo-positive samples."
        ),
    )
    parser.add_argument(
        "--decisive-only",
        action="store_true",
        help=(
            "After rollout labeling, keep only decisive training samples: "
            "utility_label in {0, 2}."
        ),
    )
    parser.add_argument("--small-model-path", default=DEFAULT_SMALL_MODEL_PATH, help="Path to the small model.")
    parser.add_argument("--large-model-path", default=DEFAULT_LARGE_MODEL_PATH, help="Path to the large model.")
    parser.add_argument(
        "--large-backend",
        choices=["hf", "vllm"],
        default="hf",
        help="Backend used for the large current-chunk rollout.",
    )
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model-name", default=None)
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Total generation budget for each branch.")
    parser.add_argument("--min-chunk-tokens", type=int, default=5)
    parser.add_argument("--max-chunk-tokens", type=int, default=30)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for HF model loading.",
    )
    return parser.parse_args()


def parse_question_ids(text: Optional[str]) -> Optional[set[int]]:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def select_question_records(dataset: List[Dict], args) -> List[Dict]:
    selected = dataset
    keep_ids = parse_question_ids(args.question_ids)
    if keep_ids is not None:
        selected = [item for item in dataset if int(item["question_id"]) in keep_ids]
    else:
        start = max(args.question_offset, 0)
        end = None if args.num_questions is None else start + max(args.num_questions, 0)
        selected = dataset[start:end]
    return selected


def chunk_risk_proxy(chunk: Dict) -> float:
    entropy = scalarize_tensor(chunk.get("final_entropy"))
    if entropy is None:
        entropy = scalarize_tensor(chunk.get("mean_entropy"))
    if entropy is None:
        entropy = 0.0

    top1 = scalarize_tensor(chunk.get("final_top1_prob"))
    if top1 is None:
        top1 = 1.0

    margin = scalarize_tensor(chunk.get("final_margin"))
    if margin is None:
        margin = 1.0

    score = float(entropy)
    score += 0.5 * max(0.0, 1.0 - float(top1))
    score += 0.5 * max(0.0, 1.0 - float(margin))
    return score


def choose_candidate_chunk_ids(
    chunks: List[Dict],
    policy: str,
    candidate_count: int,
    *,
    small_is_correct: bool,
) -> List[int]:
    total = len(chunks)
    if total == 0:
        return []
    if policy == "all":
        return [int(chunk["chunk_id"]) for chunk in chunks]

    indices = {0, total - 1}
    if total <= 2:
        return sorted(indices)

    target_count = max(candidate_count, 0)
    if policy == "risk_guided_plus_ends":
        scored = []
        previous_score = None
        for idx, chunk in enumerate(chunks):
            score = chunk_risk_proxy(chunk)
            delta = 0.0 if previous_score is None else score - previous_score
            scored.append((idx, score, delta))
            previous_score = score

        # Prefer early decisive intervention on wrong questions, and sparse,
        # cautionary negatives on already-correct questions.
        scored_by_risk = sorted(scored, key=lambda item: (-item[1], item[0]))
        scored_by_delta = sorted(scored, key=lambda item: (-item[2], item[0]))

        picks = []
        for idx, _, _ in scored_by_risk[:target_count]:
            picks.append(idx)
        for idx, _, _ in scored_by_delta[: max(1, target_count // 2)]:
            picks.append(idx)

        if not small_is_correct:
            # Bias toward earlier correction anchors on wrong questions.
            picks.extend([1, max(1, total // 4), max(1, total // 3)])
        else:
            # Keep a small number of mid-trajectory negatives for correct questions.
            picks.extend([max(1, total // 3)])

        for idx in picks:
            indices.add(min(max(idx, 0), total - 1))
        return sorted(indices)

    for rank in range(1, target_count + 1):
        rel = rank / float(target_count + 1)
        idx = int(round(rel * (total - 1)))
        indices.add(min(max(idx, 0), total - 1))
    return sorted(indices)


def build_candidate_row(record: Dict, chunk_index: int) -> Dict:
    chunks = record["chunks"]
    chunk = chunks[chunk_index]
    previous_prefix = "" if chunk_index == 0 else str(chunks[chunk_index - 1].get("prefix_text", "") or "")
    current_prefix = str(chunk.get("prefix_text", "") or "")

    return {
        "question_id": int(record["question_id"]),
        "question": record["question"],
        "ground_truth_final_answer": record.get("ground_truth_final_answer"),
        "small_final_answer": record.get("model_final_answer"),
        "small_is_correct": bool(record.get("is_final_correct", False)),
        "total_chunks": len(chunks),
        "chunk_id": int(chunk["chunk_id"]),
        "candidate_chunk_id": int(chunk["chunk_id"]),
        "relative_position": float(chunk_index / max(len(chunks) - 1, 1)),
        "prefix_before_current_chunk": previous_prefix,
        "prefix_after_current_chunk_small": current_prefix,
        "small_chunk_text_reference": str(chunk.get("chunk_text", "") or ""),
        "small_chunk_token_count_reference": int(chunk.get("token_count", chunk.get("generated_token_count", 0)) or 0),
        "chunk_cut_reason_reference": str(chunk.get("cut_reason", "") or ""),
        "source_meta": record.get("source_meta", {}),
        "boundary_hidden_state": chunk.get("boundary_hidden_state"),
        "mean_hidden_state": chunk.get("mean_hidden_state"),
        "final_entropy": chunk.get("final_entropy"),
        "final_top1_prob": chunk.get("final_top1_prob"),
        "final_margin": chunk.get("final_margin"),
        "mean_entropy": chunk.get("mean_entropy"),
        "reference_confidence": {
            "final_entropy": scalarize_tensor(chunk.get("final_entropy")),
            "final_top1_prob": scalarize_tensor(chunk.get("final_top1_prob")),
            "final_margin": scalarize_tensor(chunk.get("final_margin")),
            "mean_entropy": scalarize_tensor(chunk.get("mean_entropy")),
        },
        "label_metadata": {
            "comparison_mode": "SLM_chunk+SLM_rest vs LLM_chunk+SLM_rest",
            "label": None,
            "utility_label": None,
            "llm_preferred": None,
        },
    }


def scalarize_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return None


def load_models_for_rollouts(args):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    large_tokenizer = small_tokenizer
    large_model = None
    if args.large_backend == "hf":
        large_tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, local_files_only=True)
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map="auto",
            local_files_only=True,
        )
        large_model.eval()
    return small_model, small_tokenizer, large_model, large_tokenizer


def continue_with_small_model(
    *,
    small_model,
    small_tokenizer,
    question: str,
    prefix: str,
    args,
):
    current_prefix = prefix
    total_generated_tokens = 0
    generated_chunks: List[Dict] = []
    reached_eos = False

    while total_generated_tokens < args.max_new_tokens:
        remaining_budget = max(args.max_new_tokens - total_generated_tokens, 1)
        chunk_result = run_chunk(
            model=small_model,
            tokenizer=small_tokenizer,
            question=question,
            assistant_prefix=current_prefix,
            max_new_tokens=remaining_budget,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
            system_prompt=args.system_prompt,
            answer_type=args.answer_type,
        )
        if chunk_result["generated_token_count"] == 0:
            reached_eos = bool(chunk_result["reached_eos"])
            break
        generated_chunks.append(
            {
                "chunk_text": chunk_result["chunk_text"],
                "generated_token_count": int(chunk_result["generated_token_count"]),
                "cut_reason": chunk_result["cut_reason"],
                "reached_eos": bool(chunk_result["reached_eos"]),
            }
        )
        current_prefix = chunk_result["full_reasoning"]
        total_generated_tokens += int(chunk_result["generated_token_count"])
        reached_eos = bool(chunk_result["reached_eos"])
        if reached_eos:
            break

    return {
        "full_reasoning": current_prefix,
        "generated_token_count": total_generated_tokens,
        "generated_chunks": generated_chunks,
        "reached_eos": reached_eos,
    }


def build_rollout_args(args):
    return SimpleNamespace(
        large_handoff_chunks=1,
        max_new_tokens=args.max_new_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        system_prompt=args.system_prompt,
        answer_type=args.answer_type,
        large_backend=args.large_backend,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_model_name=args.vllm_model_name,
        vllm_timeout=args.vllm_timeout,
        large_model_path=args.large_model_path,
    )


def evaluate_branch(final_reasoning: str, ground_truth_final_answer, answer_extractor, answer_type: str):
    final_answer, has_answer = answer_extractor(final_reasoning)
    if not has_answer:
        final_answer = ""
    is_correct = has_answer and check_answer_correctness(final_answer, ground_truth_final_answer, answer_type)
    return {
        "final_reasoning": final_reasoning,
        "final_answer": final_answer,
        "is_correct": bool(is_correct),
    }


def label_candidate_rows(rows: List[Dict], args) -> List[Dict]:
    small_model, small_tokenizer, large_model, large_tokenizer = load_models_for_rollouts(args)
    rollout_args = build_rollout_args(args)
    answer_extractor = get_answer_extractor(args.answer_type)

    labeled_rows = []
    skipped_empty_current_chunk = 0
    for row in rows:
        question = row["question"]
        prefix_before = row["prefix_before_current_chunk"]
        ground_truth = row["ground_truth_final_answer"]

        small_chunk = run_chunk(
            model=small_model,
            tokenizer=small_tokenizer,
            question=question,
            assistant_prefix=prefix_before,
            max_new_tokens=args.max_new_tokens,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
            system_prompt=args.system_prompt,
            answer_type=args.answer_type,
        )
        small_branch = continue_with_small_model(
            small_model=small_model,
            small_tokenizer=small_tokenizer,
            question=question,
            prefix=small_chunk["full_reasoning"],
            args=args,
        )
        small_eval = evaluate_branch(
            small_branch["full_reasoning"],
            ground_truth,
            answer_extractor,
            args.answer_type,
        )

        large_chunk = run_large_handoff(
            model=large_model,
            tokenizer=large_tokenizer,
            question=question,
            assistant_prefix=prefix_before,
            args=rollout_args,
            num_chunks=1,
            max_total_new_tokens=args.max_new_tokens,
        )

        llm_current_chunk_text = large_chunk.get("chunks", [{}])[0].get("chunk_text", "") if large_chunk.get("chunks") else ""
        llm_current_chunk_tokens = int(large_chunk.get("generated_token_count", 0))
        if args.skip_empty_current_chunk and (
            int(small_chunk["generated_token_count"]) == 0
            or llm_current_chunk_tokens == 0
        ):
            skipped_empty_current_chunk += 1
            continue

        llm_branch = continue_with_small_model(
            small_model=small_model,
            small_tokenizer=small_tokenizer,
            question=question,
            prefix=large_chunk["full_reasoning"],
            args=args,
        )
        llm_eval = evaluate_branch(
            llm_branch["full_reasoning"],
            ground_truth,
            answer_extractor,
            args.answer_type,
        )

        llm_preferred = bool(llm_eval["is_correct"] and not small_eval["is_correct"])
        if llm_eval["is_correct"] and not small_eval["is_correct"]:
            utility_label = 2
            hard_label = "LLM"
        elif small_eval["is_correct"] and not llm_eval["is_correct"]:
            utility_label = 0
            hard_label = "SLM"
        else:
            utility_label = 1
            hard_label = "SLM"

        enriched = dict(row)
        enriched["comparison"] = {
            "small_current_chunk": {
                "chunk_text": small_chunk["chunk_text"],
                "generated_token_count": int(small_chunk["generated_token_count"]),
                "cut_reason": small_chunk["cut_reason"],
            },
            "llm_current_chunk": {
                "chunk_text": llm_current_chunk_text,
                "generated_token_count": llm_current_chunk_tokens,
                "generated_chunks": int(large_chunk.get("generated_chunks", 0)),
            },
            "small_branch": {
                "final_answer": small_eval["final_answer"],
                "is_correct": small_eval["is_correct"],
                "total_generated_tokens": int(small_chunk["generated_token_count"] + small_branch["generated_token_count"]),
            },
            "llm_branch": {
                "final_answer": llm_eval["final_answer"],
                "is_correct": llm_eval["is_correct"],
                "total_generated_tokens": int(large_chunk["generated_token_count"] + llm_branch["generated_token_count"]),
            },
        }
        enriched["label_metadata"] = {
            "comparison_mode": "SLM_chunk+SLM_rest vs LLM_chunk+SLM_rest",
            "label": hard_label,
            "utility_label": utility_label,
            "llm_preferred": llm_preferred,
        }
        enriched["label"] = 1 if hard_label == "LLM" else 0
        enriched["utility_label"] = int(utility_label)
        enriched["llm_preferred"] = bool(llm_preferred)
        labeled_rows.append(enriched)

    if skipped_empty_current_chunk:
        print(
            f"Skipped {skipped_empty_current_chunk} rollout samples because the current chunk "
            f"re-generated as empty in at least one branch."
        )

    if args.decisive_only:
        decisive_rows = [row for row in labeled_rows if int(row["utility_label"]) in {0, 2}]
        print(
            f"Keeping decisive-only rows: {len(decisive_rows)}/{len(labeled_rows)} "
            f"(utility_label in {{0, 2}})."
        )
        labeled_rows = decisive_rows
    return labeled_rows


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().to(torch.float32)
        if flat.numel() == 1:
            return float(flat.reshape(-1)[0].item())
        return flat.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_rows(rows: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".pt"):
        torch.save(rows, output_path)
        return
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    args.answer_type = resolve_answer_type(args.dataset_name, args.answer_type or DEFAULT_ANSWER_TYPE)
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)

    print(f"Loading trajectories from: {args.trajectory_path}")
    dataset = torch.load(args.trajectory_path, weights_only=False)
    selected_records = select_question_records(dataset, args)
    print(f"Selected questions: {len(selected_records)}")

    rows: List[Dict] = []
    for record in selected_records:
        chunks = record.get("chunks", [])
        if not chunks:
            continue
        per_question_candidate_count = args.candidate_count
        if bool(record.get("is_final_correct", False)):
            if args.correct_question_candidate_count is not None:
                per_question_candidate_count = args.correct_question_candidate_count
        else:
            if args.wrong_question_candidate_count is not None:
                per_question_candidate_count = args.wrong_question_candidate_count

        candidate_chunk_ids = choose_candidate_chunk_ids(
            chunks,
            args.candidate_policy,
            per_question_candidate_count,
            small_is_correct=bool(record.get("is_final_correct", False)),
        )
        for chunk_id in candidate_chunk_ids:
            rows.append(build_candidate_row(record, chunk_id))

    print(f"Built candidate rows: {len(rows)}")
    if args.label_with_rollouts:
        print("Running dual-branch rollout labeling for current-chunk decode choice...")
        rows = label_candidate_rows(rows, args)

    save_rows(rows, args.output_path)
    print(f"Saved decode-choice dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
