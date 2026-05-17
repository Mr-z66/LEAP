import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a weighted .pt training set from decode-choice JSONL labels."
    )
    parser.add_argument("--input-path", required=True, help="Path to labeled decode-choice .jsonl file.")
    parser.add_argument("--output-path", required=True, help="Path to save the .pt training dataset.")
    parser.add_argument(
        "--mode",
        default="mixed_utility",
        choices=["mixed_utility", "decisive_only"],
        help="How to map utility labels into binary labels and sample weights.",
    )
    parser.add_argument(
        "--gray-weight",
        type=float,
        default=0.35,
        help="Sample weight assigned to utility_label=1 rows in mixed_utility mode.",
    )
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=1.0,
        help="Base sample weight assigned to decisive positive rows (utility_label=2).",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=1.0,
        help="Base sample weight assigned to decisive negative rows (utility_label=0).",
    )
    parser.add_argument(
        "--drop-missing-utility",
        action="store_true",
        help="Drop rows whose utility_label is missing instead of failing.",
    )
    return parser.parse_args()


def derive_training_fields(row, args):
    meta = row.get("label_metadata", {})
    utility_label = meta.get("utility_label")
    hard_label = meta.get("label")

    if utility_label is None:
        if args.drop_missing_utility:
            return None
        raise ValueError(
            f"Missing utility_label for question_id={row.get('question_id')} candidate_chunk_id={row.get('candidate_chunk_id')}"
        )

    record = dict(row)
    record["candidate_chunk_id"] = int(record["candidate_chunk_id"])
    record["chunk_id"] = int(record["candidate_chunk_id"])
    record["utility_label"] = int(utility_label)

    if args.mode == "decisive_only":
        if utility_label not in {0, 2}:
            return None
        record["label"] = 1 if hard_label == "LLM" else 0
        record["sample_weight"] = args.positive_weight if record["label"] == 1 else args.negative_weight
        return record

    # mixed_utility
    if utility_label == 2:
        record["label"] = 1
        record["sample_weight"] = args.positive_weight
    elif utility_label == 0:
        record["label"] = 0
        record["sample_weight"] = args.negative_weight
    elif utility_label == 1:
        # Use the preference-derived hard label but downweight this gray region.
        record["label"] = 1 if hard_label == "LLM" else 0
        record["sample_weight"] = float(args.gray_weight)
    else:
        raise ValueError(f"Unexpected utility_label={utility_label}")
    return record


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    kept = 0
    skipped = 0
    utility_counts = {0: 0, 1: 0, 2: 0}
    label_counts = {0: 0, 1: 0}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            cooked = derive_training_fields(raw, args)
            if cooked is None:
                skipped += 1
                continue
            kept += 1
            utility_counts[cooked["utility_label"]] = utility_counts.get(cooked["utility_label"], 0) + 1
            label_counts[cooked["label"]] = label_counts.get(cooked["label"], 0) + 1
            records.append(cooked)

    torch.save(records, output_path)
    print(f"Saved mixed training dataset to: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Rows kept: {kept} | skipped: {skipped}")
    print(f"Utility distribution: {utility_counts}")
    print(f"Binary label distribution: {label_counts}")
    if args.mode == "mixed_utility":
        print(
            "Weights | positive decisive="
            f"{args.positive_weight:.3f} | negative decisive={args.negative_weight:.3f} | gray={args.gray_weight:.3f}"
        )


if __name__ == "__main__":
    main()
