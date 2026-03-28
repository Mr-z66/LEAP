import argparse
import json
import os
import random
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "judge_audit_samples_strict_quality.jsonl")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")
DEFAULT_MAX_JUDGE_TOKENS = 160
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_QUESTIONS = 14
DEFAULT_INCLUDE_RISKY_CHUNK = True
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Audit strict chunk labels for suspicious missed-error questions.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to strict labeled chunk data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="JSONL path for audit samples and judge output.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the audit judge model.")
    parser.add_argument("--max-judge-tokens", type=int, default=DEFAULT_MAX_JUDGE_TOKENS, help="Max new tokens for the audit judge.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Seed for deterministic sampling.")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, help="Max suspicious questions to audit.")
    parser.add_argument(
        "--include-risky-chunk",
        action="store_true",
        default=DEFAULT_INCLUDE_RISKY_CHUNK,
        help="Also audit the chunk with the highest current probe error score when available.",
    )
    parser.add_argument("--resume", action="store_true", help="Append only missing (question_id, chunk_id) pairs if output exists.")
    return parser.parse_args()


def extract_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def clamp_confidence(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric_value))


def parse_audit_response(raw_text):
    parsed = {
        "is_prefix_still_valid": 1,
        "drift_detected": 0,
        "drift_type": "unknown",
        "confidence": 0.5,
        "reason": raw_text.strip(),
        "earliest_suspicious_step": "",
        "parse_status": "fallback",
    }

    json_blob = extract_json_object(raw_text)
    if json_blob is None:
        return parsed

    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError:
        return parsed

    def to_binary(value, default):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                return 1
            if normalized in {"0", "false", "no"}:
                return 0
            return default
        if value in {0, 1}:
            return int(value)
        return default

    return {
        "is_prefix_still_valid": to_binary(payload.get("is_prefix_still_valid", 1), 1),
        "drift_detected": to_binary(payload.get("drift_detected", 0), 0),
        "drift_type": str(payload.get("drift_type", "unknown")).strip() or "unknown",
        "confidence": clamp_confidence(payload.get("confidence", 0.5)),
        "reason": str(payload.get("reason", "")).strip(),
        "earliest_suspicious_step": str(payload.get("earliest_suspicious_step", "")).strip(),
        "parse_status": "json",
    }


def build_audit_prompt(question, prefix_text):
    return f"""You are a careful math reasoning auditor.
Your job is to decide whether the student's reasoning prefix is still on a valid path.
Be more sensitive than a normal correctness judge: if the prefix already introduces a wrong intermediate value,
wrong equation, wrong relationship, wrong interpretation, or subtle logical drift that would likely lead to a wrong final answer,
you should mark drift_detected = 1 even if the final answer has not yet appeared.

Question:
{question}

Student reasoning prefix:
{prefix_text}

Audit rules:
1. Return is_prefix_still_valid = 1 only if the prefix remains on a plausible correct path.
2. Return drift_detected = 1 if the prefix already contains a likely mathematical or logical drift.
3. Incomplete reasoning is not automatically drift.
4. A small typo that does not affect the reasoning path is not drift.
5. If the prefix already commits to a wrong quantity, wrong relation, wrong sub-result, or wrong direction, mark drift.

Return JSON only:
{{
  "is_prefix_still_valid": 0 or 1,
  "drift_detected": 0 or 1,
  "drift_type": "none" | "arithmetic" | "logic" | "semantic" | "setup" | "unknown",
  "confidence": a number between 0 and 1,
  "reason": "one short sentence",
  "earliest_suspicious_step": "short phrase or empty string"
}}"""


def load_existing_pairs(output_path, resume):
    if not resume or not os.path.exists(output_path):
        return set()
    pairs = set()
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pairs.add((int(record["question_id"]), int(record["chunk_id"])))
    return pairs


def load_dataset(path):
    print(f"Loading strict labeled chunk data from: {path}")
    dataset = torch.load(path)
    if not dataset:
        raise ValueError("The strict labeled dataset is empty.")
    return dataset


def group_question_chunks(dataset):
    grouped = defaultdict(list)
    for row in dataset:
        grouped[int(row["question_id"])].append(row)
    return {
        question_id: sorted(chunks, key=lambda item: int(item["chunk_id"]))
        for question_id, chunks in grouped.items()
    }


def first_error_chunk_id(chunks):
    for chunk in chunks:
        if int(chunk["label"]) == 0:
            return int(chunk["chunk_id"])
    return None


def choose_audit_chunks(chunks, include_risky_chunk):
    if not chunks:
        return []

    candidate_ids = set()
    candidate_ids.add(int(chunks[0]["chunk_id"]))

    if len(chunks) > 2:
        candidate_ids.add(int(chunks[len(chunks) // 2]["chunk_id"]))
        candidate_ids.add(int(chunks[-2]["chunk_id"]))
    elif len(chunks) > 1:
        candidate_ids.add(int(chunks[-1]["chunk_id"]))

    if include_risky_chunk:
        risky_chunk = max(
            chunks,
            key=lambda item: float(1.0 - float(item.get("prefix_correct_score", 0.0))),
        )
        candidate_ids.add(int(risky_chunk["chunk_id"]))

    return [chunk for chunk in chunks if int(chunk["chunk_id"]) in candidate_ids]


def main():
    args = parse_args()
    random.seed(args.random_seed)

    dataset = load_dataset(args.input_path)
    question_to_chunks = group_question_chunks(dataset)

    suspicious_question_ids = []
    for question_id, chunks in sorted(question_to_chunks.items()):
        sample = chunks[0]
        if bool(sample.get("is_final_correct", False)):
            continue
        if first_error_chunk_id(chunks) is not None:
            continue
        suspicious_question_ids.append(question_id)

    suspicious_question_ids = suspicious_question_ids[: args.max_questions]
    print(f"Suspicious questions selected: {len(suspicious_question_ids)}")

    print(f"Loading audit judge model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    existing_pairs = load_existing_pairs(args.output_path, args.resume)
    audit_rows = []
    for question_id in suspicious_question_ids:
        chunks = question_to_chunks[question_id]
        audit_rows.extend(choose_audit_chunks(chunks, args.include_risky_chunk))

    print(f"Audit chunk samples selected: {len(audit_rows)}")

    with open(args.output_path, "a" if args.resume else "w", encoding="utf-8") as handle:
        for chunk in tqdm(audit_rows, desc="Auditing strict label quality"):
            question_id = int(chunk["question_id"])
            chunk_id = int(chunk["chunk_id"])
            if (question_id, chunk_id) in existing_pairs:
                continue

            prompt = build_audit_prompt(
                question=chunk["question"],
                prefix_text=chunk["prefix_text"],
            )
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_judge_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            raw_response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            parsed = parse_audit_response(raw_response)

            record = {
                "question_id": question_id,
                "chunk_id": chunk_id,
                "question": chunk["question"],
                "prefix_text": chunk["prefix_text"],
                "chunk_text": chunk.get("chunk_text", ""),
                "token_count": int(chunk.get("token_count", 0)),
                "cut_reason": chunk.get("cut_reason", "unknown"),
                "original_label": int(chunk["label"]),
                "original_judge_confidence": float(chunk.get("judge_confidence", 0.5)),
                "original_judge_error_type": chunk.get("judge_error_type", "unknown"),
                "original_judge_reason": chunk.get("judge_reason", ""),
                "original_judge_parse_status": chunk.get("judge_parse_status", "unknown"),
                "is_final_correct": bool(chunk.get("is_final_correct", False)),
                "small_final_answer": chunk.get("model_final_answer"),
                "ground_truth_final_answer": chunk.get("ground_truth_final_answer"),
                "probe_error_score": float(1.0 - float(chunk.get("prefix_correct_score", 0.0))),
                "audit_is_prefix_still_valid": parsed["is_prefix_still_valid"],
                "audit_drift_detected": parsed["drift_detected"],
                "audit_drift_type": parsed["drift_type"],
                "audit_confidence": parsed["confidence"],
                "audit_reason": parsed["reason"],
                "audit_earliest_suspicious_step": parsed["earliest_suspicious_step"],
                "audit_parse_status": parsed["parse_status"],
                "audit_raw_response": raw_response,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_pairs.add((question_id, chunk_id))

    print(f"Audit results written to: {args.output_path}")


if __name__ == "__main__":
    main()

