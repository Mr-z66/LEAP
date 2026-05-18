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
    parser.add_argument(
        "--gray-policy",
        default="all",
        choices=["all", "wrong_only", "high_risk_only", "wrong_or_high_risk"],
        help="Which utility=1 gray samples to retain in mixed_utility mode.",
    )
    parser.add_argument(
        "--gray-risk-quantile",
        type=float,
        default=0.7,
        help="Quantile threshold used by high-risk gray retention policies.",
    )
    return parser.parse_args()


def scalar_or_default(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def row_risk_score(row):
    confidence = row.get("reference_confidence", {}) or {}
    final_entropy = scalar_or_default(confidence.get("final_entropy", row.get("final_entropy", 0.0)))
    final_top1 = scalar_or_default(confidence.get("final_top1_prob", row.get("final_top1_prob", 1.0)), default=1.0)
    final_margin = scalar_or_default(confidence.get("final_margin", row.get("final_margin", 1.0)), default=1.0)
    mean_entropy = scalar_or_default(confidence.get("mean_entropy", row.get("mean_entropy", 0.0)))
    return final_entropy + mean_entropy + (1.0 - final_top1) + (1.0 - final_margin)


def keep_gray_row(row, args, gray_risk_threshold):
    if args.gray_policy == "all":
        return True

    small_is_correct = bool(row.get("small_is_correct", False))
    is_wrong_question = not small_is_correct
    is_high_risk = row_risk_score(row) >= gray_risk_threshold

    if args.gray_policy == "wrong_only":
        return is_wrong_question
    if args.gray_policy == "high_risk_only":
        return is_high_risk
    if args.gray_policy == "wrong_or_high_risk":
        return is_wrong_question or is_high_risk
    raise ValueError(f"Unsupported gray policy: {args.gray_policy}")


def derive_training_fields(row, args, gray_risk_threshold):
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
        if not keep_gray_row(record, args, gray_risk_threshold):
            return None
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
    gray_threshold = float("inf")

    raw_rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw_rows.append(json.loads(line))

    if args.mode == "mixed_utility" and args.gray_policy in {"high_risk_only", "wrong_or_high_risk"}:
        gray_scores = [row_risk_score(row) for row in raw_rows if int((row.get("label_metadata", {}) or {}).get("utility_label", -1)) == 1]
        if gray_scores:
            gray_threshold = float(torch.tensor(gray_scores, dtype=torch.float32).quantile(args.gray_risk_quantile).item())

    for raw in raw_rows:
        cooked = derive_training_fields(raw, args, gray_threshold)
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
        print(f"Gray policy: {args.gray_policy}")
        if args.gray_policy in {'high_risk_only', 'wrong_or_high_risk'} and gray_threshold != float('inf'):
            print(f"Gray risk quantile: {args.gray_risk_quantile:.3f} | threshold={gray_threshold:.6f}")
        print(
            "Weights | positive decisive="
            f"{args.positive_weight:.3f} | negative decisive={args.negative_weight:.3f} | gray={args.gray_weight:.3f}"
        )


if __name__ == "__main__":
    main()
