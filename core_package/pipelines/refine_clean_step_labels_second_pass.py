import argparse
import json
import os
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.config import MODELS
from core_package.vllm_utils import build_openai_messages, infer_served_model_name, request_vllm_chat_completion


ERROR_TYPES = {
    "none",
    "arithmetic",
    "logic",
    "setup",
    "semantic",
    "objective_mismatch",
    "unit_or_quantity",
    "unsupported_assumption",
    "format",
    "unknown",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Second-pass refinement for clean step labels. It targets questions where the small model's final "
            "answer is wrong but every prefix was labeled valid, then asks a judge to locate the earliest "
            "trajectory-level error and marks that chunk and later chunks as label=0."
        )
    )
    parser.add_argument("--input-path", required=True, help="Input clean step labeled .pt file.")
    parser.add_argument("--output-path", required=True, help="Output refined .pt file. The input is not overwritten.")
    parser.add_argument("--judge-model-path", default=MODELS.large_model_path)
    parser.add_argument("--judge-backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--max-judge-tokens", type=int, default=384)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--only-question-ids", default=None, help="Optional comma-separated question ids to refine.")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--refine-existing-errors",
        action="store_true",
        help="Also review final-wrong questions that already contain label=0. Useful for correcting first-error location.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run judging and print changes without saving.")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model-name", default=None)
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    return parser.parse_args()


def load_rows(path):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {path}")
    return rows


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(rows, path)
    print(f"Saved: {path} | rows={len(rows)}")


def group_by_question(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if "question_id" not in row or "chunk_id" not in row:
            continue
        groups[int(row["question_id"])].append((index, row))
    for question_id in list(groups):
        groups[question_id] = sorted(groups[question_id], key=lambda item: int(item[1]["chunk_id"]))
    return dict(groups)


def parse_question_id_filter(value):
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def is_false(value):
    if isinstance(value, torch.Tensor):
        return not bool(value.item())
    return not bool(value)


def candidate_questions(groups, allowed_ids=None, refine_existing_errors=False):
    candidates = []
    for question_id, indexed_chunks in sorted(groups.items()):
        if allowed_ids is not None and question_id not in allowed_ids:
            continue
        chunks = [row for _, row in indexed_chunks]
        if not chunks:
            continue
        if not is_false(chunks[0].get("is_final_correct", False)):
            continue
        labels = [int(chunk.get("label", -1)) for chunk in chunks]
        if 0 in labels and not refine_existing_errors:
            continue
        if not any(label == 1 for label in labels):
            continue
        candidates.append(question_id)
    return candidates


def compact_chunks(chunks, max_chars_per_chunk=900):
    lines = []
    for chunk in chunks:
        text = str(chunk.get("chunk_text", "")).strip()
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 20].rstrip() + " ... [truncated]"
        lines.append(f"[chunk_id={int(chunk['chunk_id'])}]\n{text}")
    return "\n\n".join(lines)


def build_second_pass_prompt(chunks):
    first = chunks[0]
    return f"""You are auditing labels for a process-routing math reasoning dataset.

The small model's final answer is wrong, but the first-pass judge labeled every reasoning prefix as valid.
Your job is to find the earliest chunk where the trajectory first becomes wrong or misaligned with the question.

Question:
{first.get("question", "")}

Reference answer text:
{first.get("ground_truth_answer_text", "")}

Reference final answer:
{first.get("ground_truth_final_answer", "")}

Small-model final answer:
{first.get("model_final_answer", "")}

Full small-model reasoning split into chunks:
{compact_chunks(chunks)}

Labeling policy:
- Separate three notions:
  1. earliest_semantic_error_chunk_id: the first chunk where the reasoning misreads the task, changes event order, uses the wrong quantity/unit/objective, chooses the wrong setup, or introduces an unsupported assumption.
  2. earliest_computational_error_chunk_id: the first chunk where an arithmetic, algebraic, comparison, or equation-manipulation error appears.
  3. recommended_training_error_chunk_id: the first chunk that should become label=0 for training a process-risk probe.
- For LEAP training, prefer the earliest semantic/setup/objective error when it exists, because the scheduler should learn to trigger before downstream computations inherit the wrong trajectory.
- If the first real error is a wrong comparison/conclusion after correct calculations, use that comparison/conclusion chunk.
- Do not mark a chunk wrong merely because it is incomplete, verbose, redundant, or has not reached the final answer yet.
- Do not mark a locally correct setup as wrong just because a later chunk will misuse it.
- "Every second item costs X%" usually means each even-numbered item is discounted, not that prices recursively decay.
- In return-trip/distance problems, distinguish distance traveled from remaining distance to the origin.
- If you genuinely cannot locate an error, return all three ids as -1 and error_type = "unknown".

Return JSON only:
{{
  "earliest_semantic_error_chunk_id": an integer chunk_id or -1,
  "earliest_computational_error_chunk_id": an integer chunk_id or -1,
  "recommended_training_error_chunk_id": an integer chunk_id or -1,
  "error_type": "arithmetic" | "logic" | "setup" | "semantic" | "objective_mismatch" | "unit_or_quantity" | "unsupported_assumption" | "format" | "unknown",
  "confidence": a number between 0 and 1,
  "reason": "one short sentence explaining why the recommended chunk is the first training error"
}}"""


def extract_json_object(raw_text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text or ""):
        try:
            payload, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def clamp_confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_second_pass_response(raw_text):
    payload = extract_json_object(raw_text)
    if payload is None:
        return {
            "earliest_error_chunk_id": -1,
            "earliest_semantic_error_chunk_id": -1,
            "earliest_computational_error_chunk_id": -1,
            "recommended_training_error_chunk_id": -1,
            "error_type": "unknown",
            "confidence": 0.0,
            "reason": (raw_text or "").strip(),
            "parse_status": "fallback",
        }
    semantic = to_int(payload.get("earliest_semantic_error_chunk_id", -1), default=-1)
    computational = to_int(payload.get("earliest_computational_error_chunk_id", -1), default=-1)
    recommended = to_int(payload.get("recommended_training_error_chunk_id", payload.get("earliest_error_chunk_id", -1)), default=-1)
    earliest = recommended
    error_type = str(payload.get("error_type", "unknown")).strip() or "unknown"
    if error_type not in ERROR_TYPES:
        error_type = "unknown"
    return {
        "earliest_error_chunk_id": earliest,
        "earliest_semantic_error_chunk_id": semantic,
        "earliest_computational_error_chunk_id": computational,
        "recommended_training_error_chunk_id": recommended,
        "error_type": error_type,
        "confidence": clamp_confidence(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason", "")).strip(),
        "parse_status": "json",
    }


def to_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def judge_second_pass(prompt, tokenizer, model, args):
    if args.judge_backend == "hf":
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_judge_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    response = request_vllm_chat_completion(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model_name=infer_served_model_name(args.judge_model_path, args.vllm_model_name),
        messages=build_openai_messages(None, prompt),
        max_tokens=args.max_judge_tokens,
        timeout=args.vllm_timeout,
    )
    return response["text"].strip()


def load_hf_judge(args):
    if args.judge_backend != "hf":
        print(
            "Using vLLM judge: "
            f"{args.vllm_base_url} | model={infer_served_model_name(args.judge_model_path, args.vllm_model_name)}"
        )
        return None, None
    print(f"Loading judge model: {args.judge_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def apply_monotone_error_labels(indexed_chunks, judge_result, raw_response, min_confidence):
    chunks = [row for _, row in indexed_chunks]
    valid_chunk_ids = {int(chunk["chunk_id"]) for chunk in chunks}
    earliest = int(judge_result["earliest_error_chunk_id"])
    if earliest not in valid_chunk_ids:
        return 0
    if judge_result["parse_status"] != "json" or judge_result["confidence"] < min_confidence:
        for _, chunk in indexed_chunks:
            chunk["second_pass_reviewed"] = True
            chunk["second_pass_earliest_error_chunk_id"] = earliest
            chunk["second_pass_earliest_semantic_error_chunk_id"] = judge_result["earliest_semantic_error_chunk_id"]
            chunk["second_pass_earliest_computational_error_chunk_id"] = judge_result["earliest_computational_error_chunk_id"]
            chunk["second_pass_recommended_training_error_chunk_id"] = judge_result["recommended_training_error_chunk_id"]
            chunk["second_pass_error_type"] = judge_result["error_type"]
            chunk["second_pass_confidence"] = judge_result["confidence"]
            chunk["second_pass_reason"] = judge_result["reason"]
            chunk["second_pass_raw_response"] = raw_response
            chunk["second_pass_parse_status"] = judge_result["parse_status"]
        return 0

    changed = 0
    for _, chunk in indexed_chunks:
        chunk_id = int(chunk["chunk_id"])
        chunk["second_pass_reviewed"] = True
        chunk["second_pass_earliest_error_chunk_id"] = earliest
        chunk["second_pass_earliest_semantic_error_chunk_id"] = judge_result["earliest_semantic_error_chunk_id"]
        chunk["second_pass_earliest_computational_error_chunk_id"] = judge_result["earliest_computational_error_chunk_id"]
        chunk["second_pass_recommended_training_error_chunk_id"] = judge_result["recommended_training_error_chunk_id"]
        chunk["second_pass_error_type"] = judge_result["error_type"]
        chunk["second_pass_confidence"] = judge_result["confidence"]
        chunk["second_pass_reason"] = judge_result["reason"]
        chunk["second_pass_raw_response"] = raw_response
        chunk["second_pass_parse_status"] = judge_result["parse_status"]
        if chunk_id >= earliest and int(chunk.get("label", -1)) != 0:
            chunk["original_label_before_second_pass"] = int(chunk.get("label", -1))
            chunk["original_judge_error_type_before_second_pass"] = str(chunk.get("judge_error_type", ""))
            chunk["original_judge_reason_before_second_pass"] = str(chunk.get("judge_reason", ""))
            chunk["label"] = 0
            chunk["label_source"] = "second_pass_monotone_repair"
            chunk["judge_error_type"] = judge_result["error_type"]
            chunk["judge_reason"] = judge_result["reason"]
            chunk["judge_confidence"] = judge_result["confidence"]
            changed += 1
    return changed


def restore_second_pass_repairs(indexed_chunks):
    restored = 0
    for _, chunk in indexed_chunks:
        if chunk.get("label_source") != "second_pass_monotone_repair":
            continue
        if "original_label_before_second_pass" not in chunk:
            continue
        original_label = chunk.get("original_label_before_second_pass")
        if original_label is None:
            continue
        chunk["label"] = int(original_label)
        chunk["label_source"] = "judge"
        if "original_judge_error_type_before_second_pass" in chunk:
            chunk["judge_error_type"] = str(chunk.get("original_judge_error_type_before_second_pass", ""))
        if "original_judge_reason_before_second_pass" in chunk:
            chunk["judge_reason"] = str(chunk.get("original_judge_reason_before_second_pass", ""))
        restored += 1
    return restored


def is_actionable_judge_result(indexed_chunks, judge_result, min_confidence):
    valid_chunk_ids = {int(chunk["chunk_id"]) for _, chunk in indexed_chunks}
    earliest = int(judge_result["earliest_error_chunk_id"])
    return (
        earliest in valid_chunk_ids
        and judge_result["parse_status"] == "json"
        and judge_result["confidence"] >= min_confidence
    )


def main():
    args = parse_args()
    rows = load_rows(args.input_path)
    groups = group_by_question(rows)
    allowed_ids = parse_question_id_filter(args.only_question_ids)
    candidates = candidate_questions(
        groups,
        allowed_ids=allowed_ids,
        refine_existing_errors=args.refine_existing_errors,
    )
    if args.max_questions is not None:
        candidates = candidates[: args.max_questions]
    print(f"Loaded rows={len(rows)} questions={len(groups)} candidates={len(candidates)}")

    tokenizer, model = load_hf_judge(args)
    total_changed = 0
    reviewed = 0
    unresolved = 0

    for question_id in tqdm(candidates, desc="Second-pass label refinement"):
        indexed_chunks = groups[question_id]
        restored = 0
        chunks = [row for _, row in indexed_chunks]
        prompt = build_second_pass_prompt(chunks)
        raw_response = judge_second_pass(prompt, tokenizer, model, args)
        judge_result = parse_second_pass_response(raw_response)
        if args.refine_existing_errors and is_actionable_judge_result(
            indexed_chunks,
            judge_result,
            min_confidence=args.low_confidence_threshold,
        ):
            restored = restore_second_pass_repairs(indexed_chunks)
        changed = apply_monotone_error_labels(
            indexed_chunks,
            judge_result,
            raw_response,
            min_confidence=args.low_confidence_threshold,
        )
        reviewed += 1
        total_changed += changed
        if changed == 0:
            unresolved += 1
        print(
            json.dumps(
                {
                    "question_id": question_id,
                    "earliest_error_chunk_id": judge_result["earliest_error_chunk_id"],
                    "earliest_semantic_error_chunk_id": judge_result["earliest_semantic_error_chunk_id"],
                    "earliest_computational_error_chunk_id": judge_result["earliest_computational_error_chunk_id"],
                    "recommended_training_error_chunk_id": judge_result["recommended_training_error_chunk_id"],
                    "error_type": judge_result["error_type"],
                    "confidence": judge_result["confidence"],
                    "restored_chunks": restored,
                    "changed_chunks": changed,
                    "reason": judge_result["reason"],
                },
                ensure_ascii=True,
            )
        )

    print(json.dumps({"reviewed": reviewed, "changed_chunks": total_changed, "unresolved": unresolved}, indent=2))
    if not args.dry_run:
        save_rows(rows, args.output_path)


if __name__ == "__main__":
    main()
