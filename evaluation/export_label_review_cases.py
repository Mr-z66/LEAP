import argparse
import json
import os
from collections import defaultdict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export selected question chunks from a labeled .pt file for manual label review.")
    parser.add_argument("--label-path", required=True, help="Input labeled .pt file.")
    parser.add_argument("--question-ids", required=True, help="Comma-separated question ids to export.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path.")
    parser.add_argument("--context-chunks", type=int, default=2, help="Chunks before and after first error to include.")
    parser.add_argument("--full", action="store_true", help="Export all chunks for each selected question.")
    parser.add_argument("--text-tail-chars", type=int, default=1200, help="Prefix tail chars to include per chunk.")
    return parser.parse_args()


def load_rows(path):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list payload in {path}")
    return rows


def group_by_question(rows):
    groups = defaultdict(list)
    for row in rows:
        if "question_id" not in row or "chunk_id" not in row:
            continue
        groups[int(row["question_id"])].append(row)
    for question_id in list(groups):
        groups[question_id] = sorted(groups[question_id], key=lambda item: int(item["chunk_id"]))
    return dict(groups)


def parse_question_ids(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def first_error_index(chunks):
    for idx, chunk in enumerate(chunks):
        if int(chunk.get("label", -1)) == 0:
            return idx
    return None


def select_chunks(chunks, context_chunks, full):
    if full:
        return chunks
    first_error = first_error_index(chunks)
    if first_error is None:
        return chunks[: min(len(chunks), 4)]
    start = max(0, first_error - context_chunks)
    end = min(len(chunks), first_error + context_chunks + 1)
    return chunks[start:end]


def compact_chunk(chunk, text_tail_chars):
    prefix = str(chunk.get("prefix_text", "") or "")
    return {
        "chunk_id": int(chunk.get("chunk_id", -1)),
        "label": int(chunk.get("label", -1)),
        "label_source": str(chunk.get("label_source", "")),
        "chunk_text": str(chunk.get("chunk_text", "")),
        "prefix_tail": prefix[-text_tail_chars:],
        "judge_error_type": str(chunk.get("judge_error_type", "")),
        "judge_reason": str(chunk.get("judge_reason", "")),
        "judge_confidence": chunk.get("judge_confidence"),
        "second_pass_reviewed": bool(chunk.get("second_pass_reviewed", False)),
        "second_pass_earliest_error_chunk_id": chunk.get("second_pass_earliest_error_chunk_id"),
        "second_pass_error_type": str(chunk.get("second_pass_error_type", "")),
        "second_pass_reason": str(chunk.get("second_pass_reason", "")),
        "second_pass_confidence": chunk.get("second_pass_confidence"),
        "original_label_before_second_pass": chunk.get("original_label_before_second_pass"),
        "original_judge_error_type_before_second_pass": str(chunk.get("original_judge_error_type_before_second_pass", "")),
        "original_judge_reason_before_second_pass": str(chunk.get("original_judge_reason_before_second_pass", "")),
    }


def main():
    args = parse_args()
    rows = load_rows(args.label_path)
    groups = group_by_question(rows)
    question_ids = parse_question_ids(args.question_ids)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for question_id in question_ids:
            chunks = groups.get(question_id, [])
            if not chunks:
                handle.write(json.dumps({"question_id": question_id, "error": "missing"}, ensure_ascii=True) + "\n")
                continue
            first = chunks[0]
            payload = {
                "question_id": question_id,
                "question": str(first.get("question", "")),
                "ground_truth_final_answer": str(first.get("ground_truth_final_answer", "")),
                "model_final_answer": str(first.get("model_final_answer", "")),
                "is_final_correct": bool(first.get("is_final_correct", False)),
                "label_sequence": [int(chunk.get("label", -1)) for chunk in chunks],
                "first_error_chunk_id": None if first_error_index(chunks) is None else int(chunks[first_error_index(chunks)]["chunk_id"]),
                "chunks": [
                    compact_chunk(chunk, args.text_tail_chars)
                    for chunk in select_chunks(chunks, args.context_chunks, args.full)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
