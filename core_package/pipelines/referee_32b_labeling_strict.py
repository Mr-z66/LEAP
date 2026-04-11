import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_package.config import MODELS, STRICT_LABEL

# ================= Default Configuration =================
DEFAULT_INPUT_DATA_PATH = STRICT_LABEL.input_path
DEFAULT_OUTPUT_DATA_PATH = STRICT_LABEL.output_path
DEFAULT_MODEL_PATH_32B = MODELS.large_model_path
DEFAULT_MAX_JUDGE_TOKENS = STRICT_LABEL.max_judge_tokens
DEFAULT_SAVE_EVERY = STRICT_LABEL.save_every
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Label heuristic chunks with a strict judge LLM.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_DATA_PATH, help="Path to chunked trajectory data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_DATA_PATH, help="Path to save labeled chunk data.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_32B, help="Local path to the judge model.")
    parser.add_argument("--max-judge-tokens", type=int, default=DEFAULT_MAX_JUDGE_TOKENS, help="Max new tokens for judge output.")
    parser.add_argument("--num-samples", type=int, default=None, help="Only label the first N questions.")
    parser.add_argument("--start-question", type=int, default=0, help="Start labeling from this question index.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save every N newly processed questions.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file if present.")
    parser.add_argument("--stop-after-first-error", action="store_true", help="Stop labeling later chunks once a question gets its first error label.")
    parser.add_argument("--include-reference-answer", action="store_true", help="Include the ground-truth answer in the judge prompt.")
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


def parse_judge_response(raw_text):
    parsed = {
        "is_prefix_correct": 0,
        "error_type": "unknown",
        "confidence": 0.5,
        "reason": raw_text.strip(),
        "parse_status": "fallback",
    }

    json_blob = extract_json_object(raw_text)
    if json_blob is None:
        return parsed

    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError:
        return parsed

    is_prefix_correct = payload.get("is_prefix_correct", 0)
    if isinstance(is_prefix_correct, bool):
        is_prefix_correct = int(is_prefix_correct)
    elif isinstance(is_prefix_correct, str):
        is_prefix_correct = 1 if is_prefix_correct.strip() in {"1", "true", "True"} else 0
    elif is_prefix_correct not in {0, 1}:
        is_prefix_correct = 0

    return {
        "is_prefix_correct": int(is_prefix_correct),
        "error_type": str(payload.get("error_type", "unknown")).strip() or "unknown",
        "confidence": clamp_confidence(payload.get("confidence", 0.5)),
        "reason": str(payload.get("reason", "")).strip(),
        "parse_status": "json",
    }


def build_judge_prompt(question, prefix_text, ground_truth_answer_text, include_reference_answer):
    reference_section = ""
    if include_reference_answer:
        reference_section = f"\nReference answer: {ground_truth_answer_text}\n"

    return f"""You are a strict math reasoning judge.
Evaluate whether the student's reasoning prefix is still logically correct so far.
Focus on the reasoning prefix itself instead of guessing from the final answer.
Be sensitive to early logical drift: if the prefix already introduces an incorrect intermediate result,
wrong equation, wrong variable relationship, wrong unit conversion, or a reasoning step that would
send the solution down an incorrect path, mark it as incorrect immediately.
Do not mark it incorrect only because the solution is unfinished, abbreviated, or ends mid-step.

Question:
{question}
{reference_section}
Student reasoning prefix:
{prefix_text}

Judging rules:
1. Return "is_prefix_correct": 1 only if all reasoning written so far is consistent with the problem and remains on a valid path.
2. Return "is_prefix_correct": 0 if the prefix already contains a mathematical, logical, or semantic mistake, even if the final answer has not appeared yet.
3. Treat an incorrect intermediate value, incorrect sub-conclusion, incorrect symbolic setup, or incorrect interpretation of the problem as an error.
4. Do not penalize a prefix merely because it is incomplete, ends at a chunk boundary, or has not yet finished the derivation.
5. When uncertain, prefer the stricter option only if there is already evidence of logical drift in the written prefix.

Return JSON only with the following schema:
{{
  "is_prefix_correct": 0 or 1,
  "error_type": "none" | "arithmetic" | "logic" | "hallucination" | "format" | "unknown",
  "confidence": a number between 0 and 1,
  "reason": "one short sentence"
}}"""


def load_existing_labels(output_path, resume):
    if not resume or not os.path.exists(output_path):
        return []
    print(f"Resuming from existing labels: {output_path}")
    return torch.load(output_path)


def build_processed_question_set(labeled_dataset):
    processed = set()
    for item in labeled_dataset:
        processed.add(int(item["question_id"]))
    return processed


def save_labels(labeled_dataset, output_path):
    torch.save(labeled_dataset, output_path)
    print(f"Checkpoint saved: {output_path} | labeled chunks: {len(labeled_dataset)}")


args = parse_args()

print(f"Loading chunked dataset from: {args.input_path}")
dataset = torch.load(args.input_path)

if args.num_samples is not None:
    dataset = dataset[: args.num_samples]
    print(f"Using first {len(dataset)} questions for this run.")

print(f"Loading judge model from: {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

labeled_dataset = load_existing_labels(args.output_path, args.resume)
processed_questions = build_processed_question_set(labeled_dataset)
newly_processed_questions = 0

questions_to_process = [sample for sample in dataset if int(sample["question_id"]) >= args.start_question]

for sample in tqdm(questions_to_process, desc="Labeling heuristic chunks with strict judge"):
    question_id = int(sample["question_id"])

    if question_id in processed_questions:
        continue

    question = sample["question"]
    ground_truth_answer_text = sample["ground_truth_answer_text"]

    for chunk in sample["chunks"]:
        prompt = build_judge_prompt(
            question=question,
            prefix_text=chunk["prefix_text"],
            ground_truth_answer_text=ground_truth_answer_text,
            include_reference_answer=args.include_reference_answer,
        )

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

        raw_response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        judge_result = parse_judge_response(raw_response)

        labeled_dataset.append(
            {
                "question_id": question_id,
                "chunk_id": chunk["chunk_id"],
                "question": question,
                "chunk_text": chunk["chunk_text"],
                "prefix_text": chunk["prefix_text"],
                "start_token_idx": chunk["start_token_idx"],
                "end_token_idx": chunk["end_token_idx"],
                "token_count": chunk["token_count"],
                "cut_reason": chunk["cut_reason"],
                "boundary_hidden_state": chunk["boundary_hidden_state"],
                "mean_hidden_state": chunk["mean_hidden_state"],
                "mean_entropy": chunk.get("mean_entropy"),
                "max_entropy": chunk.get("max_entropy"),
                "final_entropy": chunk.get("final_entropy"),
                "mean_top1_prob": chunk.get("mean_top1_prob"),
                "min_top1_prob": chunk.get("min_top1_prob"),
                "final_top1_prob": chunk.get("final_top1_prob"),
                "mean_margin": chunk.get("mean_margin"),
                "min_margin": chunk.get("min_margin"),
                "final_margin": chunk.get("final_margin"),
                "ground_truth_answer_text": ground_truth_answer_text,
                "ground_truth_final_answer": sample["ground_truth_final_answer"],
                "model_final_answer": sample["model_final_answer"],
                "is_final_correct": sample["is_final_correct"],
                "judge_model_path": args.model_path,
                "judge_mode": "strict_prefix_plus_reference" if args.include_reference_answer else "strict_prefix_only",
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
                "judge_parse_status": judge_result["parse_status"],
                "judge_confidence": judge_result["confidence"],
                "judge_error_type": judge_result["error_type"],
                "judge_reason": judge_result["reason"],
                "label": int(judge_result["is_prefix_correct"]),
            }
        )

        if args.stop_after_first_error and judge_result["is_prefix_correct"] == 0:
            break

    processed_questions.add(question_id)
    newly_processed_questions += 1

    if args.save_every > 0 and newly_processed_questions % args.save_every == 0:
        save_labels(labeled_dataset, args.output_path)

save_labels(labeled_dataset, args.output_path)
print(f"Completed question count in output: {len(processed_questions)}")
