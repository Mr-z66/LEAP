import argparse
import csv
import os
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "scheduler_case_exports", "threshold_0p10_all.csv")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "scheduler_failure_analysis.csv")
DEFAULT_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "scheduler_failure_summary.txt")
DEFAULT_LATE_TRIGGER_TOLERANCE = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze scheduler failures and bucket unrecovered questions.")
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Path to a scheduler case export CSV. Prefer the threshold_xxx_all.csv export.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the failure analysis CSV for manual follow-up.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Path to save a short text summary of failure buckets.",
    )
    parser.add_argument(
        "--late-trigger-tolerance",
        type=int,
        default=DEFAULT_LATE_TRIGGER_TOLERANCE,
        help="Allow trigger_chunk_id <= first_error_chunk_id + tolerance to count as on-time.",
    )
    return parser.parse_args()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def parse_optional_int(value):
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return int(float(text))


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def classify_failure(row, late_trigger_tolerance):
    triggered = parse_bool(row["triggered"])
    first_error_chunk_id = parse_optional_int(row.get("first_error_chunk_id"))
    trigger_chunk_id = parse_optional_int(row.get("trigger_chunk_id"))

    if not triggered:
        return "missed_no_trigger", None
    if first_error_chunk_id is None or trigger_chunk_id is None:
        return "triggered_but_large_failed", None

    trigger_gap = trigger_chunk_id - first_error_chunk_id
    if trigger_gap > late_trigger_tolerance:
        return "late_trigger", trigger_gap
    return "triggered_but_large_failed", trigger_gap


def analyze_rows(rows, late_trigger_tolerance):
    analysis_rows = []
    for row in rows:
        small_is_correct = parse_bool(row["small_is_correct"])
        scheduled_is_correct = parse_bool(row["scheduled_is_correct"])
        if small_is_correct or scheduled_is_correct:
            continue

        auto_bucket, trigger_gap = classify_failure(row, late_trigger_tolerance)
        analysis_rows.append(
            {
                "question_id": row["question_id"],
                "triggered": row["triggered"],
                "small_is_correct": row["small_is_correct"],
                "scheduled_is_correct": row["scheduled_is_correct"],
                "first_error_chunk_id": row.get("first_error_chunk_id"),
                "trigger_chunk_id": row.get("trigger_chunk_id"),
                "takeover_start_chunk_id": row.get("takeover_start_chunk_id"),
                "trigger_gap": "" if trigger_gap is None else trigger_gap,
                "auto_bucket": auto_bucket,
                "ground_truth_final_answer": row.get("ground_truth_final_answer", ""),
                "small_final_answer": row.get("small_final_answer", ""),
                "scheduled_final_answer": row.get("scheduled_final_answer", ""),
                "trigger_error_score": row.get("trigger_error_score", ""),
                "trigger_tail_bonus": row.get("trigger_tail_bonus", ""),
                "trigger_combined_score": row.get("trigger_combined_score", ""),
                "failure_pattern": "",
                "takeover_helpfulness": "",
                "notes": "",
                "question": row.get("question", ""),
                "takeover_full_reasoning": row.get("takeover_full_reasoning", ""),
            }
        )
    return analysis_rows


def save_analysis_csv(rows, output_path):
    fieldnames = [
        "question_id",
        "triggered",
        "small_is_correct",
        "scheduled_is_correct",
        "first_error_chunk_id",
        "trigger_chunk_id",
        "takeover_start_chunk_id",
        "trigger_gap",
        "auto_bucket",
        "ground_truth_final_answer",
        "small_final_answer",
        "scheduled_final_answer",
        "trigger_error_score",
        "trigger_tail_bonus",
        "trigger_combined_score",
        "failure_pattern",
        "takeover_helpfulness",
        "notes",
        "question",
        "takeover_full_reasoning",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows, summary_path):
    bucket_counts = Counter(row["auto_bucket"] for row in rows)
    avg_trigger_gap_values = [int(row["trigger_gap"]) for row in rows if str(row["trigger_gap"]).strip() != ""]
    avg_trigger_gap = sum(avg_trigger_gap_values) / len(avg_trigger_gap_values) if avg_trigger_gap_values else float("nan")

    lines = [
        "Scheduler failure analysis",
        "=" * 40,
        f"Unrecovered wrong questions: {len(rows)}",
        f"missed_no_trigger: {bucket_counts.get('missed_no_trigger', 0)}",
        f"late_trigger: {bucket_counts.get('late_trigger', 0)}",
        f"triggered_but_large_failed: {bucket_counts.get('triggered_but_large_failed', 0)}",
        f"Average trigger gap (when available): {avg_trigger_gap:.2f}" if avg_trigger_gap_values else "Average trigger gap (when available): nan",
    ]

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for line in lines:
        print(line)


def main():
    args = parse_args()
    rows = load_rows(args.input_path)
    analysis_rows = analyze_rows(rows, args.late_trigger_tolerance)
    save_analysis_csv(analysis_rows, args.output_path)
    save_summary(analysis_rows, args.summary_path)
    print(f"Saved failure analysis CSV to: {args.output_path}")
    print(f"Saved summary to: {args.summary_path}")


if __name__ == "__main__":
    main()

