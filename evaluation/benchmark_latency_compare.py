import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end wall-clock latency across multiple methods and "
            "summarize accuracy, relative latency, and utility-style comparison metrics."
        )
    )
    parser.add_argument("--config-path", required=True, help="JSON config describing methods to benchmark.")
    parser.add_argument("--output-path", default=None, help="Optional CSV output path for the summary table.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each command. The script reports mean/median/p90 over repeats.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not execute commands; only read the configured result files and summarize them.",
    )
    parser.add_argument(
        "--baseline-method",
        default=None,
        help="Method name used for relative latency / delta metrics. Overrides config baseline_method if provided.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def select_scheduler_summary(payload, method_cfg):
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        raise ValueError("Scheduler trace payload must be a dict or list of dicts.")
    if not payload:
        raise ValueError("Scheduler trace payload is empty.")

    summary_match = method_cfg.get("summary_match")
    if summary_match:
        for item in payload:
            if all(item.get(key) == value for key, value in summary_match.items()):
                return item
        raise ValueError(f"No scheduler summary matched {summary_match}.")

    selector = method_cfg.get("summary_selector", "best_scheduled_accuracy")
    if selector == "first":
        return payload[0]
    if selector == "best_scheduled_accuracy":
        return max(payload, key=lambda item: float(item.get("scheduled_accuracy", float("-inf"))))
    raise ValueError(f"Unsupported summary_selector: {selector}")


def extract_metrics(method_cfg):
    result_path = method_cfg.get("result_path")
    if not result_path:
        return {}

    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    result_type = method_cfg.get("result_type", "scheduler_trace_json")
    payload = load_json(path)

    if result_type == "model_only_json":
        return {
            "accuracy": maybe_float(payload.get("model_only_accuracy")),
            "questions_total": payload.get("questions_total"),
            "avg_generated_tokens": maybe_float(payload.get("avg_generated_tokens")),
        }
    if result_type == "large_only_json":
        return {
            "accuracy": maybe_float(payload.get("large_only_accuracy")),
            "questions_total": payload.get("questions_total"),
            "avg_generated_tokens": maybe_float(payload.get("avg_generated_tokens")),
        }
    if result_type == "scheduler_trace_json":
        summary = select_scheduler_summary(payload, method_cfg)
        return {
            "accuracy": maybe_float(summary.get("scheduled_accuracy")),
            "small_only_accuracy": maybe_float(summary.get("small_only_accuracy")),
            "gain_over_small": maybe_float(summary.get("scheduled_gain_over_small")),
            "trigger_rate": maybe_float(summary.get("trigger_rate")),
            "avg_handoff_count": maybe_float(summary.get("avg_handoff_count")),
            "avg_large_takeover_tokens": maybe_float(summary.get("avg_large_takeover_tokens")),
            "avg_trigger_progress": maybe_float(summary.get("avg_trigger_progress")),
            "questions_total": summary.get("questions_total"),
        }
    if result_type == "custom_json":
        metrics = {}
        for key, source_key in method_cfg.get("metric_map", {}).items():
            metrics[key] = maybe_float(payload.get(source_key))
        return metrics

    raise ValueError(f"Unsupported result_type: {result_type}")


def run_command(method_cfg):
    command = method_cfg["command"]
    cwd = method_cfg.get("cwd", ".")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in method_cfg.get("env", {}).items()})

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        shell=True,
        check=False,
    )
    end = time.perf_counter()
    duration = end - start
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed for method '{method_cfg['name']}' with exit code {completed.returncode}")
    return duration


def summarize_repeats(durations):
    if not durations:
        return {
            "total_latency_s": float("nan"),
            "mean_latency_s": float("nan"),
            "median_latency_s": float("nan"),
            "p90_latency_s": float("nan"),
        }
    return {
        "total_latency_s": float(sum(durations)),
        "mean_latency_s": float(statistics.mean(durations)),
        "median_latency_s": float(statistics.median(durations)),
        "p90_latency_s": float(percentile(durations, 0.9)),
    }


def safe_div(numerator, denominator):
    if denominator in (None, 0) or (isinstance(denominator, float) and math.isnan(denominator)):
        return float("nan")
    return numerator / denominator


def build_row(method_cfg, latency_stats, result_metrics):
    questions_total = method_cfg.get("questions_total") or result_metrics.get("questions_total")
    mean_latency_s = latency_stats["mean_latency_s"]
    median_latency_s = latency_stats["median_latency_s"]
    p90_latency_s = latency_stats["p90_latency_s"]

    return {
        "method": method_cfg["name"],
        "group": method_cfg.get("group", ""),
        "questions_total": questions_total,
        "accuracy": result_metrics.get("accuracy"),
        "small_only_accuracy": result_metrics.get("small_only_accuracy"),
        "gain_over_small": result_metrics.get("gain_over_small"),
        "trigger_rate": result_metrics.get("trigger_rate"),
        "avg_handoff_count": result_metrics.get("avg_handoff_count"),
        "avg_large_takeover_tokens": result_metrics.get("avg_large_takeover_tokens"),
        "avg_trigger_progress": result_metrics.get("avg_trigger_progress"),
        "avg_generated_tokens": result_metrics.get("avg_generated_tokens"),
        "mean_latency_s": mean_latency_s,
        "median_latency_s": median_latency_s,
        "p90_latency_s": p90_latency_s,
        "sec_per_question_mean": safe_div(mean_latency_s, questions_total),
        "sec_per_question_p90": safe_div(p90_latency_s, questions_total),
        "qps_mean": safe_div(questions_total, mean_latency_s),
    }


def enrich_relative_metrics(rows, baseline_method):
    baseline_row = next((row for row in rows if row["method"] == baseline_method), None)
    if baseline_row is None:
        raise ValueError(f"Baseline method '{baseline_method}' not found in rows.")

    base_acc = baseline_row.get("accuracy")
    base_mean = baseline_row.get("mean_latency_s")
    for row in rows:
        row["latency_ratio_vs_baseline"] = safe_div(row.get("mean_latency_s"), base_mean)
        if base_acc is None or row.get("accuracy") is None:
            row["accuracy_delta_vs_baseline"] = float("nan")
        else:
            row["accuracy_delta_vs_baseline"] = row["accuracy"] - base_acc

        extra_latency = (
            None
            if row.get("mean_latency_s") is None or base_mean is None
            else row["mean_latency_s"] - base_mean
        )
        acc_delta = row.get("accuracy_delta_vs_baseline")
        if (
            extra_latency is None
            or extra_latency <= 0
            or acc_delta is None
            or (isinstance(acc_delta, float) and math.isnan(acc_delta))
        ):
            row["gain_per_extra_second"] = float("nan")
        else:
            row["gain_per_extra_second"] = acc_delta / extra_latency


def print_table(rows, baseline_method):
    print("\nLatency benchmark summary")
    print("=" * 120)
    print(f"Baseline method: {baseline_method}")
    header = (
        f"{'method':<24} {'acc':>7} {'mean_s':>10} {'p90_s':>10} "
        f"{'sec/q':>10} {'lat_x':>8} {'d_acc':>8} {'gain/s':>10} {'trigger':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value, digits=4):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return "nan"
            return f"{value:.{digits}f}"

        print(
            f"{row['method']:<24} "
            f"{fmt(row.get('accuracy')):>7} "
            f"{fmt(row.get('mean_latency_s')):>10} "
            f"{fmt(row.get('p90_latency_s')):>10} "
            f"{fmt(row.get('sec_per_question_mean')):>10} "
            f"{fmt(row.get('latency_ratio_vs_baseline'), 3):>8} "
            f"{fmt(row.get('accuracy_delta_vs_baseline'), 4):>8} "
            f"{fmt(row.get('gain_per_extra_second'), 4):>10} "
            f"{fmt(row.get('trigger_rate'), 4):>8}"
        )


def write_csv(rows, output_path):
    fieldnames = [
        "method",
        "group",
        "questions_total",
        "accuracy",
        "small_only_accuracy",
        "gain_over_small",
        "trigger_rate",
        "avg_handoff_count",
        "avg_large_takeover_tokens",
        "avg_trigger_progress",
        "avg_generated_tokens",
        "mean_latency_s",
        "median_latency_s",
        "p90_latency_s",
        "sec_per_question_mean",
        "sec_per_question_p90",
        "qps_mean",
        "latency_ratio_vs_baseline",
        "accuracy_delta_vs_baseline",
        "gain_per_extra_second",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    config = load_json(args.config_path)
    methods = config["methods"]
    baseline_method = args.baseline_method or config.get("baseline_method")
    if not baseline_method:
        raise ValueError("A baseline_method must be provided either in the config or via --baseline-method.")

    rows = []
    for method_cfg in methods:
        print(f"\nRunning method: {method_cfg['name']}")
        durations = []
        if not args.skip_run:
            for run_idx in range(args.repeats):
                print(f"  repeat {run_idx + 1}/{args.repeats}")
                durations.append(run_command(method_cfg))

        latency_stats = summarize_repeats(durations) if durations else summarize_repeats([])
        result_metrics = extract_metrics(method_cfg)
        row = build_row(method_cfg, latency_stats, result_metrics)
        rows.append(row)

    enrich_relative_metrics(rows, baseline_method)
    print_table(rows, baseline_method)

    if args.output_path:
        write_csv(rows, args.output_path)
        print(f"\nSaved CSV summary to: {args.output_path}")


if __name__ == "__main__":
    main()
