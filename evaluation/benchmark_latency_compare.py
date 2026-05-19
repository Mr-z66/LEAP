import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "latency_benchmark.example.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "result" / "analysis_outputs" / "latency_benchmark_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end latency and summarize accuracy/behavior metrics for multiple methods."
    )
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH), help="Path to benchmark config JSON.")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="CSV output path for the aggregated benchmark summary.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of reruns per method when executing commands.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not execute commands; only read existing result files. Latency columns will be empty unless cached in config.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] * (1.0 - frac) + values[high] * frac


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_median(values):
    return statistics.median(values) if values else None


def safe_div(numer, denom):
    if numer is None or denom in (None, 0):
        return None
    return numer / denom


def run_command(method_name, command, workdir, repeats):
    latencies = []
    for i in range(repeats):
        print(f"[{method_name}] run {i + 1}/{repeats}")
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            check=False,
        )
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed for method '{method_name}' with exit code {completed.returncode}.")
        latencies.append(elapsed)
    return latencies


def summarize_observe_rollback_trace(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("Observe-and-rollback trace payload must be a non-empty list.")

    summaries = []
    for group in payload:
        rows = group.get("per_question_rows", []) or []
        total = len(rows)
        if total == 0:
            continue
        small_correct = sum(1 for row in rows if bool(row.get("small_is_correct", False)))
        scheduled_correct = sum(1 for row in rows if bool(row.get("scheduled_is_correct", False)))
        triggered = sum(1 for row in rows if bool(row.get("triggered", False)))
        handoff_counts = [float(row.get("handoff_count", 0.0)) for row in rows]
        triggered_rows = [row for row in rows if bool(row.get("triggered", False))]
        avg_trigger_progress = safe_mean(
            [float(row.get("avg_trigger_progress", 0.0)) for row in triggered_rows if row.get("avg_trigger_progress") is not None]
        )
        avg_large_takeover_tokens = safe_mean(
            [
                sum(
                    float(event.get("generated_token_count", 0.0))
                    for event in row.get("route_trace", [])
                    if event.get("event") == "large_handoff"
                )
                for row in rows
            ]
        )
        summaries.append(
            {
                "threshold": group.get("threshold"),
                "tail_bonus_weight": group.get("tail_bonus_weight"),
                "questions_total": total,
                "small_only_accuracy": small_correct / total,
                "scheduled_accuracy": scheduled_correct / total,
                "scheduled_gain_over_small": (scheduled_correct - small_correct) / total,
                "trigger_rate": triggered / total,
                "avg_handoff_count": safe_mean(handoff_counts),
                "avg_trigger_progress": avg_trigger_progress,
                "avg_large_takeover_tokens": avg_large_takeover_tokens,
            }
        )

    if not summaries:
        raise ValueError("No usable threshold groups found in observe-and-rollback trace.")

    best = max(summaries, key=lambda item: (item["scheduled_accuracy"], -float(item.get("threshold") or 0.0)))
    best["selected_from"] = "best_scheduled_accuracy"
    return best


def summarize_model_only_json(payload):
    rows = payload.get("rows", []) or []
    total = int(payload.get("questions_total") or len(rows) or 0)
    accuracy = payload.get("model_only_accuracy")
    avg_generated_tokens = payload.get("avg_generated_tokens")
    if accuracy is None and total > 0 and rows:
        accuracy = sum(1 for row in rows if bool(row.get("is_correct", False))) / total
    return {
        "questions_total": total,
        "small_only_accuracy": accuracy,
        "scheduled_accuracy": accuracy,
        "scheduled_gain_over_small": 0.0,
        "trigger_rate": 0.0,
        "avg_handoff_count": 0.0,
        "avg_trigger_progress": None,
        "avg_large_takeover_tokens": avg_generated_tokens,
        "selected_from": "model_only_json",
    }


def summarize_json_result(result_path, result_type):
    payload = load_json(result_path)
    if result_type == "observe_rollback_trace_json":
        return summarize_observe_rollback_trace(payload)
    if result_type == "model_only_json":
        return summarize_model_only_json(payload)
    raise ValueError(f"Unsupported result_type: {result_type}")


def build_method_record(method_cfg, repeats, skip_run):
    name = method_cfg["name"]
    result_path = Path(method_cfg["result_path"])
    workdir = method_cfg.get("workdir", str(PROJECT_ROOT))
    command = method_cfg.get("command")
    result_type = method_cfg["result_type"]

    latencies = []
    if not skip_run:
        if not command:
            raise ValueError(f"Method '{name}' is missing 'command' in config.")
        latencies = run_command(name, command, workdir, repeats)

    summary = summarize_json_result(result_path, result_type)
    questions_total = method_cfg.get("questions_total") or summary.get("questions_total")

    mean_latency = safe_mean(latencies)
    median_latency = safe_median(latencies)
    p90_latency = percentile(latencies, 0.9)

    return {
        "name": name,
        "result_type": result_type,
        "result_path": str(result_path),
        "questions_total": questions_total,
        "latency_mean_s": mean_latency,
        "latency_median_s": median_latency,
        "latency_p90_s": p90_latency,
        "sec_per_question_mean": safe_div(mean_latency, questions_total),
        "qps_mean": safe_div(questions_total, mean_latency),
        **summary,
    }


def attach_relative_metrics(records, baseline_name):
    baseline = None
    for record in records:
        if record["name"] == baseline_name:
            baseline = record
            break
    if baseline is None:
        raise ValueError(f"Baseline '{baseline_name}' not found in config methods.")

    base_latency = baseline.get("sec_per_question_mean")
    base_accuracy = baseline.get("scheduled_accuracy")

    for record in records:
        record["latency_ratio_vs_baseline"] = safe_div(record.get("sec_per_question_mean"), base_latency)
        if record.get("scheduled_accuracy") is None or base_accuracy is None:
            record["accuracy_delta_vs_baseline"] = None
        else:
            record["accuracy_delta_vs_baseline"] = record["scheduled_accuracy"] - base_accuracy

        extra_sec = None
        if record.get("sec_per_question_mean") is not None and base_latency is not None:
            extra_sec = record["sec_per_question_mean"] - base_latency
        if extra_sec is None or extra_sec <= 0 or record.get("accuracy_delta_vs_baseline") is None:
            record["gain_per_extra_second"] = None
        else:
            record["gain_per_extra_second"] = record["accuracy_delta_vs_baseline"] / extra_sec


def write_csv(output_path, records):
    fieldnames = [
        "name",
        "result_type",
        "result_path",
        "questions_total",
        "small_only_accuracy",
        "scheduled_accuracy",
        "scheduled_gain_over_small",
        "trigger_rate",
        "avg_handoff_count",
        "avg_trigger_progress",
        "avg_large_takeover_tokens",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "sec_per_question_mean",
        "qps_mean",
        "latency_ratio_vs_baseline",
        "accuracy_delta_vs_baseline",
        "gain_per_extra_second",
        "selected_from",
        "threshold",
        "tail_bonus_weight",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def print_summary(records, baseline_name):
    print()
    print(f"Latency benchmark summary (baseline: {baseline_name})")
    print("=" * 72)
    for record in records:
        mean_sec = record.get("latency_mean_s")
        sec_per_q = record.get("sec_per_question_mean")
        ratio = record.get("latency_ratio_vs_baseline")
        acc = record.get("scheduled_accuracy")
        delta = record.get("accuracy_delta_vs_baseline")
        gain_per_sec = record.get("gain_per_extra_second")
        print(record["name"])
        print(
            f"  accuracy={acc:.4f}" if acc is not None else "  accuracy=n/a"
        )
        print(
            f"  latency_mean_s={mean_sec:.4f} | sec_per_question={sec_per_q:.4f} | ratio_vs_baseline={ratio:.4f}"
            if mean_sec is not None and sec_per_q is not None and ratio is not None
            else "  latency=n/a (skip-run or result-only mode)"
        )
        print(
            f"  accuracy_delta_vs_baseline={delta:+.4f} | gain_per_extra_second={gain_per_sec:.6f}"
            if delta is not None and gain_per_sec is not None
            else f"  accuracy_delta_vs_baseline={delta:+.4f}" if delta is not None else "  accuracy_delta_vs_baseline=n/a"
        )
        if record.get("trigger_rate") is not None:
            print(
                f"  trigger_rate={record['trigger_rate']:.4f} | avg_handoff_count={record.get('avg_handoff_count', 0.0):.4f}"
            )
        print()


def main():
    args = parse_args()
    config = load_json(args.config_path)
    methods = config.get("methods", [])
    if not methods:
        raise ValueError("Config JSON must contain a non-empty 'methods' list.")
    baseline_name = config.get("baseline_name", methods[0]["name"])

    records = [build_method_record(method_cfg, args.repeats, args.skip_run) for method_cfg in methods]
    attach_relative_metrics(records, baseline_name)
    write_csv(args.output_path, records)
    print_summary(records, baseline_name)
    print(f"Saved benchmark summary to: {args.output_path}")


if __name__ == "__main__":
    main()
