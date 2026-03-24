import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Configuration =================
INPUT_DATA_PATH = "gsm8k_15b_hidden_states.pt"
OUTPUT_DATA_PATH = "gsm8k_labeled_training_data.pt"
MODEL_PATH_32B = os.path.join(os.getcwd(), "models", "Qwen2.5-32B")
MAX_JUDGE_TOKENS = 128
STOP_AFTER_FIRST_ERROR = False
INCLUDE_REFERENCE_ANSWER = False
# =================================================


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


def build_judge_prompt(question, prefix_text, ground_truth_answer_text):
    reference_section = ""
    if INCLUDE_REFERENCE_ANSWER:
        reference_section = f"\nReference answer: {ground_truth_answer_text}\n"

    return f"""You are a strict math reasoning judge.
Evaluate whether the student's reasoning prefix is still logically correct so far.
Focus on the reasoning prefix itself instead of guessing from the final answer.

Question:
{question}
{reference_section}
Student reasoning prefix:
{prefix_text}

Return JSON only with the following schema:
{{
  "is_prefix_correct": 0 or 1,
  "error_type": "none" | "arithmetic" | "logic" | "hallucination" | "format" | "unknown",
  "confidence": a number between 0 and 1,
  "reason": "one short sentence"
}}"""


print(f"Loading chunked dataset from: {INPUT_DATA_PATH}")
dataset = torch.load(INPUT_DATA_PATH)

print(f"Loading 32B judge from: {MODEL_PATH_32B}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_32B, local_files_only=True)
model_32b = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_32B,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model_32b.eval()

labeled_dataset = []

for sample in tqdm(dataset, desc="Labeling heuristic chunks with 32B judge"):
    question = sample["question"]
    question_id = sample["question_id"]
    ground_truth_answer_text = sample["ground_truth_answer_text"]

    for chunk in sample["chunks"]:
        prompt = build_judge_prompt(
            question=question,
            prefix_text=chunk["prefix_text"],
            ground_truth_answer_text=ground_truth_answer_text,
        )

        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(model_32b.device)

        with torch.no_grad():
            outputs = model_32b.generate(
                **inputs,
                max_new_tokens=MAX_JUDGE_TOKENS,
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
                "ground_truth_answer_text": ground_truth_answer_text,
                "ground_truth_final_answer": sample["ground_truth_final_answer"],
                "model_final_answer": sample["model_final_answer"],
                "is_final_correct": sample["is_final_correct"],
                "judge_model": "Qwen2.5-32B",
                "judge_mode": "prefix_only" if not INCLUDE_REFERENCE_ANSWER else "prefix_plus_reference",
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
                "judge_parse_status": judge_result["parse_status"],
                "judge_confidence": judge_result["confidence"],
                "judge_error_type": judge_result["error_type"],
                "judge_reason": judge_result["reason"],
                "label": int(judge_result["is_prefix_correct"]),
            }
        )

        if STOP_AFTER_FIRST_ERROR and judge_result["is_prefix_correct"] == 0:
            break

print(f"Saving {len(labeled_dataset)} labeled chunks to: {OUTPUT_DATA_PATH}")
torch.save(labeled_dataset, OUTPUT_DATA_PATH)
print("Done.")
