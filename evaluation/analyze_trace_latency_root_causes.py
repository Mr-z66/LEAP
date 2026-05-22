import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(PROJECT_ROOT, "result", "traces")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs", "trace_latency_root_causes")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze observe-and-rollback trace exports with a focus on latency root causes "
            "(probe counts, handoff loops, stop reasons)."
        )
    )
    parser.add_argument(
        "--trace-path",
        default=None,
        help="Path to an exported scheduler trace JSON. If omitted, analyze all observe_rollback_traces_*.json under --trace-dir.",
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help="Directory to scan when --trace-path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k-slowest",
        type=int,
        default=12,
        help="How many slowest questions to keep per threshold in the JSON report.",
    )
    return parser.parse_args()


def safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def safe_median(values):
    return statistics.median(values) if values else float("nan")


def percentile(values, q):
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    idx = int(math.ceil(q * len(values_sorted))) - 1
    idx = max(0, min(idx, len(values_sorted) - 1))
    return float(values_sorted[idx])


def threshold_tag(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_trace_entries(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Trace payload must be a non-empty list: {path}")
    return payload


def iter_trace_paths(args):
    if args.trace_path:
        yield args.trace_path
        return
    if not os.path.isdir(args.trace_dir):
        raise ValueError(f"--trace-dir does not exist: {args.trace_dir}")
    for name in sorted(os.listdir(args.trace_dir)):
        if not name.endswith(".json"):
            continue
        if not name.startswith("observe_rollback_traces_"):
            continue
        yield os.path.join(args.trace_dir, name)


def summarize_entry(entry, top_k_slowest):
    rows = entry.get("per_question_rows", []) or []
    latencies = [float(row.get("latency_s", float("nan"))) for row in rows]

    per_row = []
    stop_reasons = Counter()
    for row in rows:
        route_trace = row.get("route_trace", []) or []
        event_counts = Counter(ev.get("event") for ev in route_trace if isinstance(ev, dict))
        for ev_name, count in event_counts.items():
            if isinstance(ev_name, str) and ev_name.startswith("adaptive_handoff_stop"):
                stop_reasons[ev_name] += count

        per_row.append(
            {
                "question_id": int(row.get("question_id", -1)),
                "latency_s": float(row.get("latency_s", float("nan"))),
                "handoff_count": int(row.get("handoff_count", 0)),
                "small_generated_tokens": int(row.get("small_generated_tokens", 0)),
                "large_generated_tokens": int(row.get("large_generated_tokens", 0)),
                "n_events": int(len(route_trace)),
                "n_probe": int(event_counts.get("small_reentry_probe", 0)),
                "n_large_chunk": int(event_counts.get("adaptive_large_handoff_chunk", 0) + event_counts.get("large_handoff", 0)),
                "n_rollback": int(event_counts.get("small_observe_rollback", 0)),
                "n_small_accept": int(event_counts.get("small_accept", 0)),
            }
        )

    slowest = sorted(per_row, key=lambda item: item["latency_s"], reverse=True)[: max(int(top_k_slowest), 0)]
    return {
        "threshold": float(entry.get("threshold", float("nan"))),
        "questions_total": int(entry.get("questions_total", len(rows))),
        "scheduled_accuracy": float(entry.get("scheduled_accuracy", float("nan"))),
        "trigger_rate": float(entry.get("trigger_rate", float("nan"))),
        "avg_handoff_count": float(entry.get("avg_handoff_count", float("nan"))),
        "avg_param_weighted_token_cost": float(entry.get("avg_param_weighted_token_cost", float("nan"))),
        "latency_mean_s": float(entry.get("latency_mean_s", safe_mean(latencies))),
        "latency_median_s": float(entry.get("latency_median_s", safe_median(latencies))),
        "latency_p90_s": float(entry.get("latency_p90_s", percentile(latencies, 0.9))),
        "per_row_means": {
            "n_events_mean": safe_mean([r["n_events"] for r in per_row]),
            "n_probe_mean": safe_mean([r["n_probe"] for r in per_row]),
            "n_large_chunk_mean": safe_mean([r["n_large_chunk"] for r in per_row]),
            "n_rollback_mean": safe_mean([r["n_rollback"] for r in per_row]),
            "n_small_accept_mean": safe_mean([r["n_small_accept"] for r in per_row]),
        },
        "handoff_dist": dict(Counter(r["handoff_count"] for r in per_row)),
        "adaptive_stop_reasons": dict(stop_reasons),
        "slowest_questions": slowest,
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for trace_path in iter_trace_paths(args):
        entries = load_trace_entries(trace_path)
        trace_name = os.path.splitext(os.path.basename(trace_path))[0]
        report = {
            "trace_path": trace_path,
            "trace_name": trace_name,
            "threshold_reports": [],
            "stop_reason_union": {},
        }

        stop_union = Counter()
        threshold_reports = []
        for entry in entries:
            summary = summarize_entry(entry, top_k_slowest=args.top_k_slowest)
            threshold_reports.append(summary)
            stop_union.update(summary.get("adaptive_stop_reasons", {}) or {})

        report["threshold_reports"] = sorted(threshold_reports, key=lambda item: item["threshold"])
        report["stop_reason_union"] = dict(stop_union)

        report_path = os.path.join(args.output_dir, f"{trace_name}_root_causes.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)

        csv_rows = []
        for item in report["threshold_reports"]:
            row = {
                "threshold": item["threshold"],
                "questions_total": item["questions_total"],
                "scheduled_accuracy": item["scheduled_accuracy"],
                "trigger_rate": item["trigger_rate"],
                "avg_handoff_count": item["avg_handoff_count"],
                "latency_mean_s": item["latency_mean_s"],
                "latency_median_s": item["latency_median_s"],
                "latency_p90_s": item["latency_p90_s"],
                "avg_param_weighted_token_cost": item["avg_param_weighted_token_cost"],
                "n_events_mean": item["per_row_means"]["n_events_mean"],
                "n_probe_mean": item["per_row_means"]["n_probe_mean"],
                "n_large_chunk_mean": item["per_row_means"]["n_large_chunk_mean"],
                "n_rollback_mean": item["per_row_means"]["n_rollback_mean"],
            }
            csv_rows.append(row)

        csv_path = os.path.join(args.output_dir, f"{trace_name}_root_causes.csv")
        write_csv(
            csv_path,
            csv_rows,
            fieldnames=list(csv_rows[0].keys()) if csv_rows else [],
        )

        # Print a compact console summary.
        print(f"[trace] {trace_name}")
        for item in report["threshold_reports"]:
            thr = threshold_tag(item["threshold"])
            probes = item["per_row_means"]["n_probe_mean"]
            events = item["per_row_means"]["n_events_mean"]
            print(
                "  "
                f"thr={thr} "
                f"acc={item['scheduled_accuracy']:.4f} "
                f"lat_mean={item['latency_mean_s']:.2f}s "
                f"p90={item['latency_p90_s']:.2f}s "
                f"handoff={item['avg_handoff_count']:.2f} "
                f"probe={probes:.2f} "
                f"events={events:.1f}"
            )
        top_stop = ", ".join(f"{k}:{v}" for k, v in stop_union.most_common(4))
        if top_stop:
            print(f"  top_stop={top_stop}")


if __name__ == "__main__":
    main()

