#!/usr/bin/env python
"""Summarize observe-and-rollback scheduler trace JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_paths", nargs="+", help="Scheduler trace JSON files.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV output path.")
    parser.add_argument("--json-path", default=None, help="Optional JSON output path.")
    parser.add_argument("--thresholds", default=None, help="Optional comma-separated threshold filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Examples per error bucket.")
    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def avg(values: Iterable[Any]) -> Optional[float]:
    cleaned = [safe_float(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    return mean(cleaned) if cleaned else None


def med(values: Iterable[Any]) -> Optional[float]:
    cleaned = [safe_float(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    return median(cleaned) if cleaned else None


def first_trigger_event(question_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for event in question_row.get("route_trace", []) or []:
        if event.get("event") == "small_observe_rollback":
            return event
    return None


def summarize_threshold(trace_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
    questions = row.get("per_question_rows", []) or []
    triggered = [q for q in questions if q.get("triggered")]
    correct_questions = [q for q in questions if q.get("small_is_correct")]
    error_questions = [q for q in questions if not q.get("small_is_correct")]
    triggered_correct = [q for q in triggered if q.get("small_is_correct")]
    triggered_error = [q for q in triggered if not q.get("small_is_correct")]
    fixed_errors = [q for q in error_questions if q.get("scheduled_is_correct")]
    harmed_correct = [q for q in correct_questions if not q.get("scheduled_is_correct")]

    first_events = [(q, first_trigger_event(q)) for q in triggered]
    first_events = [(q, ev) for q, ev in first_events if ev is not None]
    first_reasons = Counter(str(ev.get("cut_reason", "")) for _, ev in first_events)
    handoff_counts = Counter(int(q.get("handoff_count", 0)) for q in questions)

    return {
        "trace": trace_name,
        "threshold": row.get("threshold"),
        "questions_total": row.get("questions_total"),
        "small_only_accuracy": row.get("small_only_accuracy"),
        "scheduled_accuracy": row.get("scheduled_accuracy"),
        "scheduled_gain_over_small": row.get("scheduled_gain_over_small"),
        "trigger_rate": row.get("trigger_rate"),
        "questions_triggered": row.get("questions_triggered"),
        "error_questions_total": row.get("error_questions_total"),
        "error_questions_triggered": row.get("error_questions_triggered"),
        "correct_questions_triggered": len(triggered_correct),
        "false_alarm_correct_question_rate": row.get("false_alarm_correct_question_rate"),
        "fixed_error_questions": len(fixed_errors),
        "harmed_correct_questions": len(harmed_correct),
        "avg_handoff_count": row.get("avg_handoff_count"),
        "handoff_count_hist": dict(sorted(handoff_counts.items())),
        "avg_trigger_progress": row.get("avg_trigger_progress"),
        "first_trigger_progress_median": med(ev.get("progress_ratio") for _, ev in first_events),
        "first_trigger_score_median": med(ev.get("combined_score") for _, ev in first_events),
        "first_trigger_cut_reasons": dict(first_reasons.most_common(8)),
        "avg_small_generated_tokens": row.get("avg_small_generated_tokens"),
        "avg_large_generated_tokens": row.get("avg_large_generated_tokens"),
        "avg_large_takeover_tokens": row.get("avg_large_takeover_tokens"),
        "avg_param_weighted_token_cost": row.get("avg_param_weighted_token_cost"),
        "latency_mean_s": row.get("latency_mean_s"),
        "latency_median_s": row.get("latency_median_s"),
        "latency_p90_s": row.get("latency_p90_s"),
    }


def bucket_examples(row: Dict[str, Any], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {
        "fixed_errors": [],
        "missed_errors": [],
        "harmed_correct": [],
        "false_alarm_correct_preserved": [],
    }
    for q in row.get("per_question_rows", []) or []:
        first_event = first_trigger_event(q)
        item = {
            "question_id": q.get("question_id"),
            "small_is_correct": q.get("small_is_correct"),
            "scheduled_is_correct": q.get("scheduled_is_correct"),
            "triggered": q.get("triggered"),
            "handoff_count": q.get("handoff_count"),
            "trigger_progress": q.get("avg_trigger_progress"),
            "first_trigger_score": first_event.get("combined_score") if first_event else None,
            "first_trigger_progress": first_event.get("progress_ratio") if first_event else None,
            "small_final_answer": q.get("small_final_answer"),
            "scheduled_final_answer": q.get("scheduled_final_answer"),
        }
        if not q.get("small_is_correct") and q.get("scheduled_is_correct"):
            buckets["fixed_errors"].append(item)
        elif not q.get("small_is_correct") and not q.get("scheduled_is_correct"):
            buckets["missed_errors"].append(item)
        elif q.get("small_is_correct") and not q.get("scheduled_is_correct"):
            buckets["harmed_correct"].append(item)
        elif q.get("small_is_correct") and q.get("triggered"):
            buckets["false_alarm_correct_preserved"].append(item)

    return {key: values[:top_k] for key, values in buckets.items()}


def write_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace",
        "threshold",
        "questions_total",
        "small_only_accuracy",
        "scheduled_accuracy",
        "scheduled_gain_over_small",
        "trigger_rate",
        "error_questions_triggered",
        "error_questions_total",
        "correct_questions_triggered",
        "false_alarm_correct_question_rate",
        "fixed_error_questions",
        "harmed_correct_questions",
        "avg_handoff_count",
        "first_trigger_progress_median",
        "first_trigger_score_median",
        "avg_large_generated_tokens",
        "avg_param_weighted_token_cost",
        "latency_mean_s",
        "latency_p90_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    threshold_filter = None
    if args.thresholds:
        threshold_filter = {float(part.strip()) for part in args.thresholds.split(",") if part.strip()}
    summaries: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for raw_path in args.trace_paths:
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: float("nan"))
        if not isinstance(data, list):
            raise ValueError(f"Expected trace list in {path}")
        details[str(path)] = {}
        for row in data:
            if threshold_filter is not None and float(row.get("threshold")) not in threshold_filter:
                continue
            summary = summarize_threshold(path.name, row)
            summaries.append(summary)
            details[str(path)][str(row.get("threshold"))] = {
                "summary": summary,
                "examples": bucket_examples(row, args.top_k),
            }

    for summary in summaries:
        print(
            f"{summary['trace']} | thr={summary['threshold']} | "
            f"acc={summary['scheduled_accuracy']:.4f} | gain={summary['scheduled_gain_over_small']:.4f} | "
            f"trigger={summary['trigger_rate']:.4f} | "
            f"err={summary['error_questions_triggered']}/{summary['error_questions_total']} | "
            f"false_alarm={summary['false_alarm_correct_question_rate']:.4f} | "
            f"fixed={summary['fixed_error_questions']} | harmed={summary['harmed_correct_questions']} | "
            f"cost={summary['avg_param_weighted_token_cost']:.1f} | lat={summary['latency_mean_s']:.2f}s"
        )

    if args.csv_path:
        write_csv(Path(args.csv_path), summaries)
        print(f"Wrote CSV summary: {args.csv_path}")
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(details, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Wrote JSON detail: {args.json_path}")


if __name__ == "__main__":
    main()
