import argparse
import json
import os

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_PATH = os.path.join(PROJECT_ROOT, "observe_rollback_traces_384.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_outputs")
DEFAULT_SLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B")
DEFAULT_LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot threshold-accuracy and threshold-FLOPs comparison figures.")
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH, help="Path to trace export JSON.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output figures.")
    parser.add_argument("--tail-bonus-weight", type=float, default=0.0, help="Filter results by tail bonus weight.")
    parser.add_argument("--llm-accuracy", type=float, default=None, help="Optional LLM-only accuracy baseline.")
    parser.add_argument("--slm-accuracy", type=float, default=None, help="Optional SLM-only accuracy baseline; defaults to measured small-only accuracy.")
    parser.add_argument("--llm-token-proxy", type=float, default=384.0, help="LLM-only average generated tokens per question.")
    parser.add_argument("--slm-params-b", type=float, default=1.5, help="SLM parameter size in billions, used as per-token FLOPs proxy.")
    parser.add_argument("--llm-params-b", type=float, default=32.0, help="LLM parameter size in billions, used as per-token FLOPs proxy.")
    parser.add_argument("--cost-mode", choices=["token_proxy", "approx_flops"], default="approx_flops", help="How to compute normalized cost.")
    parser.add_argument("--prompt-token-proxy", type=float, default=0.0, help="Fallback prompt token count when trace rows do not contain prompt_token_count.")
    parser.add_argument("--slm-model-path", default=DEFAULT_SLM_MODEL_PATH, help="Local SLM directory containing config.json for approximate FLOPs mode.")
    parser.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH, help="Local LLM directory containing config.json for approximate FLOPs mode.")
    parser.add_argument("--base-param-flops-factor", type=float, default=2.0, help="Linear per-token parameter FLOPs factor for approximate decode/prefill.")
    parser.add_argument("--attn-flops-factor", type=float, default=4.0, help="Attention FLOPs factor used with hidden_size and num_hidden_layers.")
    parser.add_argument("--emit-summary-json", default=None, help="Optional path to dump threshold summary JSON.")
    return parser.parse_args()


def load_trace(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Trace JSON must be a list of threshold groups.")
    return payload


def load_model_dims(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    if hidden_size is None or num_hidden_layers is None:
        raise ValueError(f"Could not find hidden_size/num_hidden_layers in {config_path}")
    return {"hidden_size": float(hidden_size), "num_hidden_layers": float(num_hidden_layers)}


def is_close(a, b, eps=1e-9):
    return abs(float(a) - float(b)) < eps


def sum_decode_attention_lengths(prefix_tokens, new_tokens):
    return new_tokens * prefix_tokens + (new_tokens * (new_tokens - 1)) / 2.0


def estimate_prefill_flops(total_tokens, params_b, model_dims, base_param_flops_factor, attn_flops_factor):
    if total_tokens <= 0:
        return 0.0
    param_term = base_param_flops_factor * params_b * 1e9 * total_tokens
    attention_term = attn_flops_factor * model_dims["num_hidden_layers"] * model_dims["hidden_size"] * (total_tokens ** 2)
    return param_term + attention_term


def estimate_decode_flops(prefix_tokens, new_tokens, params_b, model_dims, base_param_flops_factor, attn_flops_factor):
    if new_tokens <= 0:
        return 0.0
    param_term = base_param_flops_factor * params_b * 1e9 * new_tokens
    attention_term = attn_flops_factor * model_dims["num_hidden_layers"] * model_dims["hidden_size"] * sum_decode_attention_lengths(prefix_tokens, new_tokens)
    return param_term + attention_term


def per_question_cost_breakdown(
    row,
    cost_mode,
    slm_cost_ratio,
    llm_token_proxy,
    slm_params_b=None,
    llm_params_b=None,
    slm_model_dims=None,
    llm_model_dims=None,
    prompt_token_proxy=0.0,
    base_param_flops_factor=2.0,
    attn_flops_factor=4.0,
):
    prompt_tokens = float(row.get("prompt_token_count", prompt_token_proxy) or 0.0)
    committed_prefix_tokens = 0.0
    slm_decode_tokens = 0.0
    rollback_waste_tokens = 0.0
    llm_prefix_rebuild_tokens = 0.0
    llm_decode_tokens = 0.0

    slm_decode_cost = 0.0
    rollback_waste_cost = 0.0
    llm_prefix_rebuild_cost = 0.0
    llm_decode_cost = 0.0

    for event in row.get("route_trace", []):
        event_type = event.get("event")
        token_count = float(event.get("generated_token_count", 0.0))
        full_prefix_tokens = prompt_tokens + committed_prefix_tokens

        if cost_mode == "token_proxy":
            small_chunk_cost = slm_cost_ratio * token_count
            large_prefix_cost = committed_prefix_tokens
            large_decode_cost = token_count
        else:
            small_chunk_cost = estimate_decode_flops(
                prefix_tokens=full_prefix_tokens,
                new_tokens=token_count,
                params_b=slm_params_b,
                model_dims=slm_model_dims,
                base_param_flops_factor=base_param_flops_factor,
                attn_flops_factor=attn_flops_factor,
            )
            large_prefix_cost = estimate_prefill_flops(
                total_tokens=full_prefix_tokens,
                params_b=llm_params_b,
                model_dims=llm_model_dims,
                base_param_flops_factor=base_param_flops_factor,
                attn_flops_factor=attn_flops_factor,
            )
            large_decode_cost = estimate_decode_flops(
                prefix_tokens=full_prefix_tokens,
                new_tokens=token_count,
                params_b=llm_params_b,
                model_dims=llm_model_dims,
                base_param_flops_factor=base_param_flops_factor,
                attn_flops_factor=attn_flops_factor,
            )

        if event_type == "small_accept":
            slm_decode_tokens += token_count
            slm_decode_cost += small_chunk_cost
            committed_prefix_tokens += token_count
        elif event_type == "small_observe_rollback":
            slm_decode_tokens += token_count
            rollback_waste_tokens += token_count
            rollback_waste_cost += small_chunk_cost
        elif event_type == "large_handoff":
            llm_prefix_rebuild_tokens += committed_prefix_tokens
            llm_decode_tokens += token_count
            llm_prefix_rebuild_cost += large_prefix_cost
            llm_decode_cost += large_decode_cost
            committed_prefix_tokens += token_count

    llm_only_prefix_tokens = prompt_tokens
    if cost_mode == "token_proxy":
        llm_only_cost = llm_token_proxy
    else:
        llm_only_cost = estimate_prefill_flops(
            total_tokens=llm_only_prefix_tokens,
            params_b=llm_params_b,
            model_dims=llm_model_dims,
            base_param_flops_factor=base_param_flops_factor,
            attn_flops_factor=attn_flops_factor,
        ) + estimate_decode_flops(
            prefix_tokens=llm_only_prefix_tokens,
            new_tokens=llm_token_proxy,
            params_b=llm_params_b,
            model_dims=llm_model_dims,
            base_param_flops_factor=base_param_flops_factor,
            attn_flops_factor=attn_flops_factor,
        )

    return {
        "prompt_tokens": prompt_tokens,
        "slm_decode_tokens": slm_decode_tokens,
        "rollback_waste_tokens": rollback_waste_tokens,
        "llm_prefix_rebuild_tokens": llm_prefix_rebuild_tokens,
        "llm_decode_tokens": llm_decode_tokens,
        "slm_decode_cost": slm_decode_cost,
        "rollback_waste_cost": rollback_waste_cost,
        "llm_prefix_rebuild_cost": llm_prefix_rebuild_cost,
        "llm_decode_cost": llm_decode_cost,
        "llm_only_cost": llm_only_cost,
    }


def build_points(
    payload,
    tail_bonus_weight,
    cost_mode,
    slm_cost_ratio,
    llm_token_proxy,
    prompt_token_proxy=0.0,
    slm_params_b=None,
    llm_params_b=None,
    slm_model_dims=None,
    llm_model_dims=None,
    base_param_flops_factor=2.0,
    attn_flops_factor=4.0,
):
    points = []
    for group in payload:
        if not is_close(group.get("tail_bonus_weight", float("nan")), tail_bonus_weight):
            continue
        rows = group.get("per_question_rows", [])
        if not rows:
            continue

        total = len(rows)
        small_correct = sum(1 for row in rows if bool(row.get("small_is_correct", False)))
        scheduled_correct = sum(1 for row in rows if bool(row.get("scheduled_is_correct", False)))
        triggered = sum(1 for row in rows if bool(row.get("triggered", False)))
        handoff_counts = [float(row.get("handoff_count", 0.0)) for row in rows if bool(row.get("triggered", False))]

        cost_breakdowns = [
            per_question_cost_breakdown(
                row,
                cost_mode=cost_mode,
                slm_cost_ratio=slm_cost_ratio,
                llm_token_proxy=llm_token_proxy,
                slm_params_b=slm_params_b,
                llm_params_b=llm_params_b,
                slm_model_dims=slm_model_dims,
                llm_model_dims=llm_model_dims,
                prompt_token_proxy=prompt_token_proxy,
                base_param_flops_factor=base_param_flops_factor,
                attn_flops_factor=attn_flops_factor,
            )
            for row in rows
        ]
        avg_prompt_tokens = sum(item["prompt_tokens"] for item in cost_breakdowns) / total
        avg_slm_decode_tokens = sum(item["slm_decode_tokens"] for item in cost_breakdowns) / total
        avg_rollback_waste_tokens = sum(item["rollback_waste_tokens"] for item in cost_breakdowns) / total
        avg_llm_prefix_rebuild_tokens = sum(item["llm_prefix_rebuild_tokens"] for item in cost_breakdowns) / total
        avg_llm_decode_tokens = sum(item["llm_decode_tokens"] for item in cost_breakdowns) / total

        avg_slm_decode_cost = sum(item["slm_decode_cost"] for item in cost_breakdowns) / total
        avg_rollback_waste_cost = sum(item["rollback_waste_cost"] for item in cost_breakdowns) / total
        avg_llm_prefix_rebuild_cost = sum(item["llm_prefix_rebuild_cost"] for item in cost_breakdowns) / total
        avg_llm_decode_cost = sum(item["llm_decode_cost"] for item in cost_breakdowns) / total
        avg_llm_only_cost = sum(item["llm_only_cost"] for item in cost_breakdowns) / total

        overall_flops_ratio = (
            avg_slm_decode_cost
            + avg_rollback_waste_cost
            + avg_llm_prefix_rebuild_cost
            + avg_llm_decode_cost
        ) / avg_llm_only_cost
        llm_usage_ratio = avg_llm_decode_tokens / llm_token_proxy

        points.append(
            {
                "threshold": float(group["threshold"]),
                "small_only_accuracy": small_correct / total,
                "scheduled_accuracy": scheduled_correct / total,
                "gain_over_small": (scheduled_correct - small_correct) / total,
                "trigger_rate": triggered / total,
                "avg_handoff_count": (sum(handoff_counts) / len(handoff_counts)) if handoff_counts else float("nan"),
                "avg_prompt_tokens": avg_prompt_tokens,
                "avg_large_takeover_tokens": avg_llm_decode_tokens,
                "avg_small_tokens": avg_slm_decode_tokens,
                "avg_rollback_waste_tokens": avg_rollback_waste_tokens,
                "avg_llm_prefix_rebuild_tokens": avg_llm_prefix_rebuild_tokens,
                "avg_slm_decode_cost": avg_slm_decode_cost,
                "avg_rollback_waste_cost": avg_rollback_waste_cost,
                "avg_llm_prefix_rebuild_cost": avg_llm_prefix_rebuild_cost,
                "avg_large_takeover_cost": avg_llm_decode_cost,
                "avg_llm_only_cost": avg_llm_only_cost,
                "llm_usage_ratio": llm_usage_ratio,
                "overall_flops_ratio": overall_flops_ratio,
            }
        )

    points.sort(key=lambda item: item["threshold"])
    return points


def plot_accuracy(points, llm_accuracy, slm_accuracy, out_path):
    thresholds = [p["threshold"] for p in points]
    scheduled = [p["scheduled_accuracy"] for p in points]

    plt.figure(figsize=(8.8, 5.6))
    plt.plot(thresholds, scheduled, marker="o", linewidth=2.2, label="LEAP Scheduled", color="#2F5597")

    if slm_accuracy is not None:
        plt.axhline(y=slm_accuracy, linestyle="--", linewidth=1.8, color="#4C78A8", label=f"SLM-only ({slm_accuracy:.3f})")
    if llm_accuracy is not None:
        plt.axhline(y=llm_accuracy, linestyle="--", linewidth=1.8, color="#F58518", label=f"LLM-only ({llm_accuracy:.3f})")

    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold (No Tail Bonus)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_flops_ratio(points, out_path):
    thresholds = [p["threshold"] for p in points]
    leap_ratio = [p["overall_flops_ratio"] for p in points]

    plt.figure(figsize=(8.8, 5.6))
    plt.plot(thresholds, leap_ratio, marker="o", linewidth=2.2, color="#E45756", label="LEAP total FLOPs ratio")
    plt.axhline(y=1.0, linestyle="--", linewidth=1.6, color="#F58518", label="LLM-only (1.0)")
    plt.xlabel("Threshold")
    plt.ylabel("Total FLOPs Ratio (relative to LLM-only)")
    plt.title("Total FLOPs Ratio vs Threshold (No Tail Bonus)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_accuracy_flops_frontier(points, llm_accuracy, out_path):
    x = [p["overall_flops_ratio"] for p in points]
    y = [p["scheduled_accuracy"] for p in points]
    labels = [f"{p['threshold']:.2f}" for p in points]

    plt.figure(figsize=(8.8, 5.6))
    plt.plot(x, y, marker="o", linewidth=2.0, color="#2F5597", label="LEAP thresholds")
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)

    if llm_accuracy is not None:
        plt.scatter([1.0], [llm_accuracy], color="#F58518", s=80, marker="s", label=f"LLM-only ({llm_accuracy:.3f})")

    plt.xlabel("Total FLOPs Ratio (relative to LLM-only)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy-FLOPs Tradeoff Frontier (No Tail Bonus)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    if args.llm_params_b <= 0.0:
        raise ValueError("--llm-params-b must be positive.")
    if args.llm_token_proxy <= 0.0:
        raise ValueError("--llm-token-proxy must be positive.")

    slm_cost_ratio = args.slm_params_b / args.llm_params_b
    slm_model_dims = None
    llm_model_dims = None
    if args.cost_mode == "approx_flops":
        slm_model_dims = load_model_dims(args.slm_model_path)
        llm_model_dims = load_model_dims(args.llm_model_path)

    payload = load_trace(args.trace_path)
    points = build_points(
        payload,
        tail_bonus_weight=args.tail_bonus_weight,
        cost_mode=args.cost_mode,
        slm_cost_ratio=slm_cost_ratio,
        llm_token_proxy=args.llm_token_proxy,
        prompt_token_proxy=args.prompt_token_proxy,
        slm_params_b=args.slm_params_b,
        llm_params_b=args.llm_params_b,
        slm_model_dims=slm_model_dims,
        llm_model_dims=llm_model_dims,
        base_param_flops_factor=args.base_param_flops_factor,
        attn_flops_factor=args.attn_flops_factor,
    )
    if not points:
        raise ValueError(
            f"No threshold groups found for tail_bonus_weight={args.tail_bonus_weight}. "
            f"Check --trace-path and --tail-bonus-weight."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    measured_slm_acc = points[0]["small_only_accuracy"]
    slm_acc = args.slm_accuracy if args.slm_accuracy is not None else measured_slm_acc

    acc_path = os.path.join(args.output_dir, "threshold_accuracy_compare_no_tail.png")
    flops_path = os.path.join(args.output_dir, "threshold_flops_compare_no_tail.png")
    frontier_path = os.path.join(args.output_dir, "accuracy_flops_frontier_no_tail.png")
    summary_path = args.emit_summary_json or os.path.join(args.output_dir, "threshold_summary_no_tail.json")

    plot_accuracy(points, llm_accuracy=args.llm_accuracy, slm_accuracy=slm_acc, out_path=acc_path)
    plot_flops_ratio(points, out_path=flops_path)
    plot_accuracy_flops_frontier(points, llm_accuracy=args.llm_accuracy, out_path=frontier_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)

    print("Saved figures:")
    print(f"  {acc_path}")
    print(f"  {flops_path}")
    print(f"  {frontier_path}")
    print(f"Saved threshold summary: {summary_path}")
    print(f"Cost mode: {args.cost_mode}")

    print("\nThreshold summary (tail bonus filtered):")
    for p in points:
        print(
            f"threshold={p['threshold']:.2f} | "
            f"scheduled_acc={p['scheduled_accuracy']:.4f} | "
            f"gain={p['gain_over_small']:+.4f} | "
            f"trigger={p['trigger_rate']:.4f} | "
            f"avg_prompt_tokens={p['avg_prompt_tokens']:.2f} | "
            f"avg_small_tokens={p['avg_small_tokens']:.2f} | "
            f"avg_rollback_waste={p['avg_rollback_waste_tokens']:.2f} | "
            f"avg_llm_prefix_rebuild={p['avg_llm_prefix_rebuild_tokens']:.2f} | "
            f"avg_large_tokens={p['avg_large_takeover_tokens']:.2f} | "
            f"llm_usage_ratio={p['llm_usage_ratio']:.4f} | "
            f"overall_flops_ratio={p['overall_flops_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
