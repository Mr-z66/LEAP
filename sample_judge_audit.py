import json
import os
import random

import torch

# ================= Configuration =================
DATA_PATH = os.path.join(os.getcwd(), "gsm8k_labeled_training_data.pt")
OUTPUT_PATH = os.path.join(os.getcwd(), "judge_audit_samples.jsonl")
NUM_SAMPLES = 100
RANDOM_SEED = 42
# =================================================


print(f"Loading labeled chunk dataset from: {DATA_PATH}")
dataset = torch.load(DATA_PATH)

if not dataset:
    raise ValueError("The labeled dataset is empty.")

random.seed(RANDOM_SEED)
sample_size = min(NUM_SAMPLES, len(dataset))
audit_samples = random.sample(dataset, sample_size)

with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
    for sample in audit_samples:
        record = {
            "question_id": int(sample["question_id"]),
            "chunk_id": int(sample["chunk_id"]),
            "question": sample["question"],
            "chunk_text": sample["chunk_text"],
            "prefix_text": sample["prefix_text"],
            "cut_reason": sample.get("cut_reason", "unknown"),
            "token_count": int(sample["token_count"]),
            "label": int(sample["label"]),
            "judge_confidence": float(sample.get("judge_confidence", 0.5)),
            "judge_error_type": sample.get("judge_error_type", "unknown"),
            "judge_reason": sample.get("judge_reason", ""),
            "judge_parse_status": sample.get("judge_parse_status", "unknown"),
            "is_final_correct": bool(sample.get("is_final_correct", False)),
            "human_label": None,
            "human_note": "",
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Exported {sample_size} audit samples to: {OUTPUT_PATH}")
