import argparse
import csv
import glob
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(root: Path):
    rows = []
    glimp_summary = root / "glimprouter_benchmark_summary.json"
    rsd_summary = root / "rsd_benchmark_summary.json"

    if glimp_summary.exists():
        for item in load_json(glimp_summary):
            rows.append(
                {
                    "method": "glimprouter",
                    "run_root": str(root),
                    "dataset_name": item.get("dataset_name"),
                    "questions_total": item.get("questions_total"),
                    "accuracy": item.get("accuracy"),
                    "threshold": item.get("score_threshold"),
                    "latency_mean_s": item.get("latency_mean_s"),
                    "latency_median_s": item.get("latency_median_s"),
                    "latency_p90_s": item.get("latency_p90_s"),
                    "avg_small_tokens": item.get("avg_small_tokens"),
                    "avg_large_tokens": item.get("avg_large_tokens"),
                    "avg_router_calls": item.get("avg_score_calls"),
                    "avg_large_fraction": item.get("avg_large_step_fraction"),
                    "avg_param_weighted_token_cost": item.get("avg_param_weighted_token_cost"),
                }
            )

    if rsd_summary.exists():
        for item in load_json(rsd_summary):
            rows.append(
                {
                    "method": "rsd",
                    "run_root": str(root),
                    "dataset_name": item.get("dataset_name"),
                    "questions_total": item.get("questions_total"),
                    "accuracy": item.get("accuracy"),
                    "threshold": item.get("prm_threshold"),
                    "latency_mean_s": item.get("latency_mean_s"),
                    "latency_median_s": item.get("latency_median_s"),
                    "latency_p90_s": item.get("latency_p90_s"),
                    "avg_small_tokens": item.get("avg_draft_tokens"),
                    "avg_large_tokens": item.get("avg_target_tokens"),
                    "avg_router_calls": item.get("avg_prm_score_calls"),
                    "avg_large_fraction": item.get("avg_target_step_fraction"),
                    "avg_param_weighted_token_cost": item.get("avg_param_weighted_token_cost"),
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize LEAP baseline reproduction outputs.")
    parser.add_argument("--roots", nargs="+", required=True, help="Baseline output roots or glob patterns.")
    parser.add_argument("--output-csv", default="result/baselines/fair_repro_summary.csv")
    parser.add_argument("--output-json", default="result/baselines/fair_repro_summary.json")
    args = parser.parse_args()

    roots = []
    for pattern in args.roots:
        matches = glob.glob(pattern)
        roots.extend(Path(match) for match in matches)
    roots = sorted({root for root in roots if root.exists()})

    rows = []
    for root in roots:
        rows.extend(collect_rows(root))

    fieldnames = [
        "method",
        "dataset_name",
        "questions_total",
        "accuracy",
        "threshold",
        "avg_param_weighted_token_cost",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "avg_small_tokens",
        "avg_large_tokens",
        "avg_router_calls",
        "avg_large_fraction",
        "run_root",
    ]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote JSON to {output_json}")


if __name__ == "__main__":
    main()

