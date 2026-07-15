"""Summarize SVAMP controller ablations with question-level bootstrap CIs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DISPLAY_ORDER = [
    "no_intervention",
    "no_rollback",
    "slm_repair",
    "no_return",
    "fixed1",
    "fixed2",
    "fixed4",
    "adaptive2to4",
    "fixed4_h1",
    "fixed4_h4",
    "fixed4_no_cooldown",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--large-only-cost", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=55)
    return parser.parse_args()


def load_trace(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected one threshold group in {path}, got {len(payload)}")
    return payload[0]


def percentile_interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_mean(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(samples, dtype=np.float64)
    for idx in range(samples):
        sample_ids = rng.integers(0, len(values), len(values))
        draws[idx] = values[sample_ids].mean()
    return percentile_interval(draws)


def paired_bootstrap_delta(values, reference, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(values) != len(reference):
        raise ValueError("Paired bootstrap requires equal-length aligned arrays")
    delta = values - reference
    return bootstrap_mean(delta, samples, rng)


def fmt_ci(value, lo, hi, scale=1.0, digits=2):
    return f"{value * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def main():
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    traces = {}
    for path in trace_dir.glob("*.json"):
        traces[path.stem] = load_trace(path)
    if "fixed4" not in traces:
        raise ValueError("The fixed4 reference trace is required for paired comparisons")

    reference_rows = {str(row["question_id"]): row for row in traces["fixed4"]["per_question_rows"]}
    reference_ids = list(reference_rows)
    reference_acc = np.asarray(
        [float(bool(reference_rows[qid]["scheduled_is_correct"])) for qid in reference_ids]
    )
    reference_cost = np.asarray(
        [float(reference_rows[qid]["param_weighted_token_cost"]) for qid in reference_ids]
    )

    output_rows = []
    for order, name in enumerate(DISPLAY_ORDER):
        if name not in traces:
            continue
        trace = traces[name]
        rows = {str(row["question_id"]): row for row in trace["per_question_rows"]}
        if set(rows) != set(reference_ids):
            raise ValueError(f"Question IDs for {name} do not match fixed4")
        acc = np.asarray([float(bool(rows[qid]["scheduled_is_correct"])) for qid in reference_ids])
        cost = np.asarray([float(rows[qid]["param_weighted_token_cost"]) for qid in reference_ids])
        rng = np.random.default_rng(args.seed + order * 1009)
        acc_lo, acc_hi = bootstrap_mean(acc, args.bootstrap_samples, rng)
        cost_lo, cost_hi = bootstrap_mean(cost / args.large_only_cost, args.bootstrap_samples, rng)
        delta_acc_lo, delta_acc_hi = paired_bootstrap_delta(
            acc, reference_acc, args.bootstrap_samples, rng
        )
        delta_cost_lo, delta_cost_hi = paired_bootstrap_delta(
            cost / args.large_only_cost,
            reference_cost / args.large_only_cost,
            args.bootstrap_samples,
            rng,
        )
        output_rows.append(
            {
                "controller": name,
                "questions": len(acc),
                "accuracy": float(acc.mean()),
                "accuracy_ci_low": acc_lo,
                "accuracy_ci_high": acc_hi,
                "relative_decode_cost": float(cost.mean() / args.large_only_cost),
                "relative_cost_ci_low": cost_lo,
                "relative_cost_ci_high": cost_hi,
                "accuracy_delta_vs_fixed4": float((acc - reference_acc).mean()),
                "accuracy_delta_ci_low": delta_acc_lo,
                "accuracy_delta_ci_high": delta_acc_hi,
                "cost_delta_vs_fixed4": float(
                    (cost / args.large_only_cost - reference_cost / args.large_only_cost).mean()
                ),
                "cost_delta_ci_low": delta_cost_lo,
                "cost_delta_ci_high": delta_cost_hi,
                "trigger_rate": float(trace["trigger_rate"]),
                "avg_handoff_count": float(trace["avg_handoff_count"]),
                "avg_small_discarded_tokens": float(trace["avg_small_discarded_tokens"]),
                "avg_large_generated_tokens": float(trace["avg_large_generated_tokens"]),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# SVAMP controller ablation",
        "",
        f"Question-level bootstrap: {args.bootstrap_samples} samples, seed {args.seed}.",
        "",
        "| Controller | Accuracy % [95% CI] | Relative decode cost [95% CI] | ΔAcc vs fixed4 % [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            "| {controller} | {acc} | {cost} | {delta} |".format(
                controller=row["controller"],
                acc=fmt_ci(
                    row["accuracy"], row["accuracy_ci_low"], row["accuracy_ci_high"], scale=100
                ),
                cost=fmt_ci(
                    row["relative_decode_cost"],
                    row["relative_cost_ci_low"],
                    row["relative_cost_ci_high"],
                    digits=3,
                ),
                delta=fmt_ci(
                    row["accuracy_delta_vs_fixed4"],
                    row["accuracy_delta_ci_low"],
                    row["accuracy_delta_ci_high"],
                    scale=100,
                ),
            )
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

