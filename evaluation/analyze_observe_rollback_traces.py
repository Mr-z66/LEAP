import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_PATH = os.path.join(
    PROJECT_ROOT,
    "result",
    "traces",
    "observe_rollback_traces_math500_vllm_hidden_only_t2048.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs", "observe_rollback_trace_analysis")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze observe-and-rollback trace exports.")
    parser.add_argument(
        "--trace-path",
        default=DEFAULT_TRACE_PATH,
        help="Path to the exported scheduler trace JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold to analyze. If omitted, pick the threshold with the best scheduled gain over small.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many example question ids to keep per bucket in the summary JSON.",
    )
    return parser.parse_args()


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def safe_median(values):
    return statistics.median(values) if values else float("nan")


def threshold_tag(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_trace_entries(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Trace payload must be a non-empty list: {path}")
    return payload


def pick_entry(entries, threshold):
    if threshold is None:
        return max(entries, key=lambda item: scheduled_gain(item))

    for entry in entries:
        if math.isclose(float(entry["threshold"]), float(threshold), rel_tol=0.0, abs_tol=1e-9):
            return entry
    available = ", ".join(str(item["threshold"]) for item in entries)
    raise ValueError(f"Threshold {threshold} not found. Available thresholds: {available}")


def scheduled_gain(entry):
    rows = entry["per_question_rows"]
    total = len(rows)
    small_correct = sum(1 for row in rows if row["small_is_correct"])
    scheduled_correct = sum(1 for row in rows if row["scheduled_is_correct"])
    return (scheduled_correct - small_correct) / max(total, 1)


def first_trigger_event(row):
    for event in row.get("route_trace", []):
        if event.get("event") == "small_observe_rollback":
            return event
    return None


def first_large_handoff(row):
    for event in row.get("route_trace", []):
        if event.get("event") == "large_handoff":
            return event
    return None


def first_large_chunk_text(row):
    handoff = first_large_handoff(row)
    if not handoff:
        return ""
    chunks = handoff.get("chunks", [])
    if not chunks:
        return ""
    return str(chunks[0].get("chunk_text", "") or "")


def classify_row(row):
    small_is_correct = bool(row["small_is_correct"])
    scheduled_is_correct = bool(row["scheduled_is_correct"])
    triggered = bool(row["triggered"])

    if not small_is_correct and scheduled_is_correct:
        return "rescued"
    if small_is_correct and not scheduled_is_correct:
        return "harmed"
    if not small_is_correct and not scheduled_is_correct and triggered:
        return "stubborn_wrong"
    if small_is_correct and scheduled_is_correct and triggered:
        return "correct_triggered"
    if small_is_correct and scheduled_is_correct and not triggered:
        return "correct_untriggered"
    if not small_is_correct and not scheduled_is_correct and not triggered:
        return "missed_wrong"
    return "other"


def flatten_row(row):
    trigger_event = first_trigger_event(row)
    first_large_chunk = first_large_chunk_text(row)
    trigger_text = ""
    trigger_chunk_id = ""
    trigger_score = ""
    trigger_progress = ""
    trigger_cut_reason = ""
    if trigger_event is not None:
        trigger_text = str(trigger_event.get("chunk_text", "") or "")
        trigger_chunk_id = trigger_event.get("chunk_id", "")
        trigger_score = trigger_event.get("combined_score", "")
        trigger_progress = trigger_event.get("progress_ratio", "")
        trigger_cut_reason = trigger_event.get("cut_reason", "")

    scheduled_final_answer = row.get("scheduled_final_answer")
    small_final_answer = row.get("small_final_answer")
    scheduled_text = "" if scheduled_final_answer is None else str(scheduled_final_answer)
    small_text = "" if small_final_answer is None else str(small_final_answer)

    return {
        "question_id": row["question_id"],
        "category": classify_row(row),
        "small_is_correct": row["small_is_correct"],
        "scheduled_is_correct": row["scheduled_is_correct"],
        "triggered": row["triggered"],
        "handoff_count": row["handoff_count"],
        "avg_trigger_score": row["avg_trigger_score"],
        "avg_trigger_progress": row["avg_trigger_progress"],
        "small_final_answer": small_text,
        "scheduled_final_answer": scheduled_text,
        "answer_changed": small_text != scheduled_text,
        "scheduled_answer_length": len(scheduled_text),
        "first_trigger_chunk_id": trigger_chunk_id,
        "first_trigger_score": trigger_score,
        "first_trigger_progress": trigger_progress,
        "first_trigger_cut_reason": trigger_cut_reason,
        "first_trigger_chunk_text": trigger_text,
        "first_large_chunk_text": first_large_chunk,
        "route_trace_event_count": len(row.get("route_trace", [])),
    }


def summarize_rows(flat_rows, threshold):
    bucket_counter = Counter(row["category"] for row in flat_rows)
    triggered_rows = [row for row in flat_rows if row["triggered"]]
    wrong_rows = [row for row in flat_rows if not row["small_is_correct"]]
    rescued_rows = [row for row in flat_rows if row["category"] == "rescued"]
    harmed_rows = [row for row in flat_rows if row["category"] == "harmed"]
    stubborn_rows = [row for row in flat_rows if row["category"] == "stubborn_wrong"]

    changed_wrong = [
        row for row in stubborn_rows
        if row["small_final_answer"] != row["scheduled_final_answer"]
    ]
    unchanged_wrong = [
        row for row in stubborn_rows
        if row["small_final_answer"] == row["scheduled_final_answer"]
    ]

    return {
        "threshold": threshold,
        "questions_total": len(flat_rows),
        "triggered_questions": len(triggered_rows),
        "trigger_rate": len(triggered_rows) / max(len(flat_rows), 1),
        "small_wrong_questions": len(wrong_rows),
        "bucket_counts": dict(bucket_counter),
        "rescued_count": len(rescued_rows),
        "harmed_count": len(harmed_rows),
        "stubborn_wrong_count": len(stubborn_rows),
        "changed_but_still_wrong_count": len(changed_wrong),
        "unchanged_still_wrong_count": len(unchanged_wrong),
        "avg_trigger_progress": safe_mean([row["avg_trigger_progress"] for row in triggered_rows]),
        "median_trigger_progress": safe_median([row["avg_trigger_progress"] for row in triggered_rows]),
        "avg_rescued_trigger_progress": safe_mean([row["avg_trigger_progress"] for row in rescued_rows if row["triggered"]]),
        "avg_harmed_trigger_progress": safe_mean([row["avg_trigger_progress"] for row in harmed_rows if row["triggered"]]),
        "avg_stubborn_trigger_progress": safe_mean([row["avg_trigger_progress"] for row in stubborn_rows if row["triggered"]]),
        "handoff_count_distribution": dict(Counter(row["handoff_count"] for row in triggered_rows)),
    }


def save_csv(rows, path):
    fieldnames = [
        "question_id",
        "category",
        "small_is_correct",
        "scheduled_is_correct",
        "triggered",
        "handoff_count",
        "avg_trigger_score",
        "avg_trigger_progress",
        "small_final_answer",
        "scheduled_final_answer",
        "answer_changed",
        "scheduled_answer_length",
        "first_trigger_chunk_id",
        "first_trigger_score",
        "first_trigger_progress",
        "first_trigger_cut_reason",
        "first_trigger_chunk_text",
        "first_large_chunk_text",
        "route_trace_event_count",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(summary, flat_rows, output_path, top_k):
    by_bucket = {}
    for bucket in sorted(set(row["category"] for row in flat_rows)):
        bucket_rows = [row for row in flat_rows if row["category"] == bucket]
        by_bucket[bucket] = {
            "count": len(bucket_rows),
            "example_question_ids": [row["question_id"] for row in bucket_rows[:top_k]],
            "avg_trigger_progress": safe_mean([row["avg_trigger_progress"] for row in bucket_rows if row["triggered"]]),
            "median_trigger_progress": safe_median([row["avg_trigger_progress"] for row in bucket_rows if row["triggered"]]),
        }

    payload = {
        "summary": summary,
        "bucket_examples": by_bucket,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_summary(summary, trace_path, output_dir):
    print("\nObserve-and-rollback trace analysis")
    print("=" * 50)
    print(f"Trace path: {trace_path}")
    print(f"Threshold: {summary['threshold']}")
    print(f"Questions total: {summary['questions_total']}")
    print(f"Triggered: {summary['triggered_questions']} ({summary['trigger_rate']:.4f})")
    print(f"Small wrong questions: {summary['small_wrong_questions']}")
    print(f"Rescued: {summary['rescued_count']}")
    print(f"Harmed: {summary['harmed_count']}")
    print(f"Stubborn wrong: {summary['stubborn_wrong_count']}")
    print(f"Changed but still wrong: {summary['changed_but_still_wrong_count']}")
    print(f"Unchanged still wrong: {summary['unchanged_still_wrong_count']}")
    print(f"Avg trigger progress: {summary['avg_trigger_progress']:.4f}")
    print(f"Median trigger progress: {summary['median_trigger_progress']:.4f}")
    print(f"Handoff count distribution: {summary['handoff_count_distribution']}")
    print(f"Saved outputs to: {output_dir}")


def main():
    args = parse_args()
    entries = load_trace_entries(args.trace_path)
    entry = pick_entry(entries, args.threshold)
    threshold = float(entry["threshold"])

    flat_rows = [flatten_row(row) for row in entry["per_question_rows"]]
    summary = summarize_rows(flat_rows, threshold)

    threshold_dir = os.path.join(args.output_dir, f"threshold_{threshold_tag(threshold)}")
    os.makedirs(threshold_dir, exist_ok=True)

    all_csv_path = os.path.join(threshold_dir, "all_cases.csv")
    save_csv(flat_rows, all_csv_path)

    for bucket in sorted(set(row["category"] for row in flat_rows)):
        bucket_rows = [row for row in flat_rows if row["category"] == bucket]
        bucket_csv_path = os.path.join(threshold_dir, f"{bucket}.csv")
        save_csv(bucket_rows, bucket_csv_path)

    summary_json_path = os.path.join(threshold_dir, "summary.json")
    save_summary(summary, flat_rows, summary_json_path, args.top_k)

    print_summary(summary, args.trace_path, threshold_dir)


if __name__ == "__main__":
    main()
