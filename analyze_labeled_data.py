import collections
import os
import statistics

import torch

# ================= Configuration =================
DATA_PATH = os.path.join(os.getcwd(), "gsm8k_labeled_training_data.pt")
# =================================================


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    return statistics.median(values) if values else 0.0


print(f"Loading labeled chunk dataset from: {DATA_PATH}")
dataset = torch.load(DATA_PATH)

if not dataset:
    raise ValueError("The labeled dataset is empty.")

label_counter = collections.Counter()
cut_reason_counter = collections.Counter()
error_type_counter = collections.Counter()
parse_status_counter = collections.Counter()
chunk_counts_per_question = collections.Counter()
first_error_chunk_positions = []
first_error_relative_positions = []
judge_confidences = []

question_to_chunks = collections.defaultdict(list)
for sample in dataset:
    question_to_chunks[sample["question_id"]].append(sample)

for question_id, chunks in question_to_chunks.items():
    chunks = sorted(chunks, key=lambda item: item["chunk_id"])
    chunk_counts_per_question[question_id] = len(chunks)

    first_error_index = None
    for chunk in chunks:
        label_counter[int(chunk["label"])] += 1
        cut_reason_counter[chunk.get("cut_reason", "unknown")] += 1
        error_type_counter[chunk.get("judge_error_type", "unknown")] += 1
        parse_status_counter[chunk.get("judge_parse_status", "unknown")] += 1
        judge_confidences.append(float(chunk.get("judge_confidence", 0.5)))

        if int(chunk["label"]) == 0 and first_error_index is None:
            first_error_index = int(chunk["chunk_id"])

    if first_error_index is not None:
        first_error_chunk_positions.append(first_error_index)
        if len(chunks) > 0:
            first_error_relative_positions.append(first_error_index / len(chunks))

num_questions = len(question_to_chunks)
num_chunks = len(dataset)
positive_chunks = label_counter.get(1, 0)
negative_chunks = label_counter.get(0, 0)

print("\nDataset overview")
print("=" * 50)
print(f"Questions: {num_questions}")
print(f"Chunks: {num_chunks}")
print(f"Prefix-correct chunks (label=1): {positive_chunks}")
print(f"Prefix-error chunks (label=0): {negative_chunks}")
print(f"Error ratio: {negative_chunks / max(1, num_chunks):.4f}")

chunk_count_values = list(chunk_counts_per_question.values())
print("\nChunking distribution")
print("=" * 50)
print(f"Avg chunks/question: {safe_mean(chunk_count_values):.2f}")
print(f"Median chunks/question: {safe_median(chunk_count_values):.2f}")
print(f"Min chunks/question: {min(chunk_count_values)}")
print(f"Max chunks/question: {max(chunk_count_values)}")
for cut_reason, count in cut_reason_counter.most_common():
    print(f"{cut_reason}: {count}")

print("\nJudge output quality")
print("=" * 50)
for parse_status, count in parse_status_counter.most_common():
    print(f"{parse_status}: {count}")
print(f"Avg judge confidence: {safe_mean(judge_confidences):.4f}")
print(f"Median judge confidence: {safe_median(judge_confidences):.4f}")

print("\nError type distribution")
print("=" * 50)
for error_type, count in error_type_counter.most_common():
    print(f"{error_type}: {count}")

print("\nFirst error position")
print("=" * 50)
questions_with_error = len(first_error_chunk_positions)
print(f"Questions with at least one error chunk: {questions_with_error}")
if questions_with_error:
    print(f"Avg first-error chunk id: {safe_mean(first_error_chunk_positions):.2f}")
    print(f"Median first-error chunk id: {safe_median(first_error_chunk_positions):.2f}")
    print(f"Avg first-error relative position: {safe_mean(first_error_relative_positions):.4f}")
    print(f"Median first-error relative position: {safe_median(first_error_relative_positions):.4f}")
