import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs", "label_audits")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit chunk-level strict-prefix labels for consistency issues.")
    parser.add_argument("--label-path", required=True, help="Path to strict labeled chunk .pt file.")
    parser.add_argument("--label-key", default="label", help="Label field. 1 means prefix-correct, 0 means risky/error.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for audit JSON/CSV/JSONL outputs.")
    parser.add_argument("--output-prefix", default=None, help="Optional output prefix.")
    parser.add_argument("--max-examples-per-bucket", type=int, default=20, help="Examples exported per suspicious bucket.")
    parser.add_argument("--text-tail-chars", type=int, default=600, help="How many prefix tail chars to include in JSONL examples.")
    return parser.parse_args()


def as_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).reshape(-1)
        return float(value[0]) if len(value) else default
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if len(arr) else default


def as_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def load_dataset(path):
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list in {path}")
    return payload


def group_by_question(dataset, label_key):
    groups = defaultdict(list)
    skipped = 0
    for row in dataset:
        if label_key not in row:
            skipped += 1
            continue
        label = int(row[label_key])
        if label not in {0, 1}:
            skipped += 1
            continue
        groups[int(row["question_id"])].append(row)
    for question_id in list(groups):
        groups[question_id] = sorted(groups[question_id], key=lambda item: int(item["chunk_id"]))
    return dict(groups), skipped


def first_error_index(chunks, label_key):
    for index, chunk in enumerate(chunks):
        if int(chunk[label_key]) == 0:
            return index
    return None


def has_zero_to_one_flip(chunks, label_key):
    seen_error = False
    for chunk in chunks:
        label = int(chunk[label_key])
        if label == 0:
            seen_error = True
        elif seen_error and label == 1:
            return True
    return False


def transition_pattern(chunks, label_key):
    labels = [int(chunk[label_key]) for chunk in chunks]
    if not labels:
        return "empty"
    transitions = []
    last = labels[0]
    for label in labels[1:]:
        if label != last:
            transitions.append(f"{last}->{label}")
            last = label
    return ",".join(transitions) if transitions else "constant"


def confidence_stats(chunks):
    values = [
        as_float(chunk.get("judge_confidence"), default=None)
        for chunk in chunks
        if chunk.get("judge_confidence") is not None
    ]
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "min": None, "low_lt_0p6": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "low_lt_0p6": int(sum(value < 0.6 for value in values)),
    }


def summarize_question(question_id, chunks, label_key):
    labels = [int(chunk[label_key]) for chunk in chunks]
    final_correct = as_bool(chunks[0].get("is_final_correct", False))
    first_error = first_error_index(chunks, label_key)
    conf = confidence_stats(chunks)
    return {
        "question_id": int(question_id),
        "chunks": int(len(chunks)),
        "final_correct": final_correct,
        "error_chunks": int(sum(1 for label in labels if label == 0)),
        "error_fraction": float(sum(1 for label in labels if label == 0) / max(len(labels), 1)),
        "first_error_chunk_id": None if first_error is None else int(chunks[first_error]["chunk_id"]),
        "first_error_index": first_error,
        "zero_to_one_flip": has_zero_to_one_flip(chunks, label_key),
        "transition_pattern": transition_pattern(chunks, label_key),
        "judge_confidence_mean": conf["mean"],
        "judge_confidence_min": conf["min"],
        "low_confidence_chunks_lt_0p6": conf["low_lt_0p6"],
    }


def bucket_examples(question_summaries):
    buckets = defaultdict(list)
    for row in question_summaries:
        if not row["final_correct"] and row["error_chunks"] == 0:
            buckets["final_wrong_but_no_error_chunk"].append(row)
        if row["final_correct"] and row["error_chunks"] > 0:
            buckets["final_correct_but_has_error_chunk"].append(row)
        if row["zero_to_one_flip"]:
            buckets["zero_to_one_label_flip"].append(row)
        if row["judge_confidence_min"] is not None and row["judge_confidence_min"] < 0.6:
            buckets["low_judge_confidence"].append(row)
    return buckets


def compact_example(question_id, chunks, label_key, text_tail_chars):
    labels = [int(chunk[label_key]) for chunk in chunks]
    first_error = first_error_index(chunks, label_key)
    chosen_index = 0 if first_error is None else first_error
    chosen = chunks[chosen_index]
    prefix = str(chosen.get("prefix_text", "") or "")
    return {
        "question_id": int(question_id),
        "chunk_id": int(chosen.get("chunk_id", chosen_index)),
        "label_sequence": labels,
        "final_correct": as_bool(chunks[0].get("is_final_correct", False)),
        "ground_truth_final_answer": str(chunks[0].get("ground_truth_final_answer", "")),
        "model_final_answer": str(chunks[0].get("model_final_answer", "")),
        "question": str(chunks[0].get("question", "")),
        "chunk_text": str(chosen.get("chunk_text", "")),
        "prefix_tail": prefix[-text_tail_chars:],
        "judge_confidence": as_float(chosen.get("judge_confidence"), default=None),
        "judge_error_type": str(chosen.get("judge_error_type", "")),
        "judge_reason": str(chosen.get("judge_reason", "")),
        "judge_raw_response": str(chosen.get("judge_raw_response", ""))[:1200],
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.label_path))[0]

    dataset = load_dataset(args.label_path)
    groups, skipped = group_by_question(dataset, args.label_key)
    question_summaries = [
        summarize_question(question_id, chunks, args.label_key)
        for question_id, chunks in sorted(groups.items())
    ]

    label_counts = Counter()
    cut_reasons = Counter()
    error_types = Counter()
    parse_statuses = Counter()
    for chunks in groups.values():
        for chunk in chunks:
            label_counts[int(chunk[args.label_key])] += 1
            cut_reasons[str(chunk.get("cut_reason", "unknown"))] += 1
            error_types[str(chunk.get("judge_error_type", "unknown"))] += 1
            parse_statuses[str(chunk.get("judge_parse_status", "unknown"))] += 1

    buckets = bucket_examples(question_summaries)
    summary = {
        "label_path": args.label_path,
        "label_key": args.label_key,
        "rows_total": len(dataset),
        "rows_used": int(sum(len(chunks) for chunks in groups.values())),
        "rows_skipped": skipped,
        "questions": len(groups),
        "label_counts": dict(label_counts),
        "error_chunk_prevalence": label_counts.get(0, 0) / max(sum(label_counts.values()), 1),
        "final_wrong_questions": int(sum(1 for row in question_summaries if not row["final_correct"])),
        "final_correct_questions": int(sum(1 for row in question_summaries if row["final_correct"])),
        "final_wrong_but_no_error_chunk": len(buckets.get("final_wrong_but_no_error_chunk", [])),
        "final_correct_but_has_error_chunk": len(buckets.get("final_correct_but_has_error_chunk", [])),
        "zero_to_one_label_flip_questions": len(buckets.get("zero_to_one_label_flip", [])),
        "low_judge_confidence_questions": len(buckets.get("low_judge_confidence", [])),
        "cut_reasons": dict(cut_reasons),
        "judge_error_types": dict(error_types),
        "judge_parse_statuses": dict(parse_statuses),
    }

    json_path = os.path.join(args.output_dir, f"{output_prefix}_label_audit_summary.json")
    csv_path = os.path.join(args.output_dir, f"{output_prefix}_question_label_audit.csv")
    jsonl_path = os.path.join(args.output_dir, f"{output_prefix}_suspicious_label_examples.jsonl")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    write_csv(csv_path, question_summaries)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for bucket_name, rows in sorted(buckets.items()):
            for row in rows[: args.max_examples_per_bucket]:
                example = compact_example(
                    row["question_id"],
                    groups[row["question_id"]],
                    args.label_key,
                    args.text_tail_chars,
                )
                example["bucket"] = bucket_name
                handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()

