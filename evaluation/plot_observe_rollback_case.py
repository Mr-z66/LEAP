import argparse
import json
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_PATH = os.path.join(PROJECT_ROOT, "result", "traces", "observe_rollback_traces.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "analysis_outputs")

EVENT_STYLES = {
    "small_accept": {"label": "SLM", "color": "#4C78A8", "text_color": "white"},
    "small_observe_rollback": {"label": "Rollback", "color": "#E45756", "text_color": "white"},
    "large_handoff": {"label": "LLM", "color": "#F58518", "text_color": "black"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a chunk-level observe-and-rollback case figure.")
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH, help="Path to observe-and-rollback trace JSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold to select from the trace file.")
    parser.add_argument("--question-id", type=int, required=True, help="Question id to visualize.")
    parser.add_argument("--output-path", default=None, help="Optional explicit PNG output path.")
    parser.add_argument("--max-text-chars", type=int, default=56, help="Max characters shown per chunk box.")
    return parser.parse_args()


def load_trace_rows(trace_path, threshold=None):
    with open(trace_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Trace JSON must be a list of threshold summaries.")
    if threshold is None:
        return payload[0]["per_question_rows"]
    for item in payload:
        if abs(float(item["threshold"]) - float(threshold)) < 1e-9:
            return item["per_question_rows"]
    raise KeyError(f"Threshold {threshold} not found in trace file.")


def shorten(text, max_chars):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def wrap_label(text, width=18):
    return "\n".join(textwrap.wrap(text, width=width))


def expand_events(route_trace):
    expanded = []
    for event in route_trace:
        if event["event"] in {"small_accept", "small_observe_rollback"}:
            expanded.append(
                {
                    "event": event["event"],
                    "title": f"c{event['chunk_id']}",
                    "subtitle": f"score={event['combined_score']:.3f}",
                    "text": event["chunk_text"],
                }
            )
        elif event["event"] == "large_handoff":
            handoff_index = event["handoff_index"]
            for chunk in event.get("chunks", []):
                expanded.append(
                    {
                        "event": "large_handoff",
                        "title": f"H{handoff_index}-c{chunk['handoff_local_chunk_id']}",
                        "subtitle": f"handoff#{handoff_index}",
                        "text": chunk["chunk_text"],
                    }
                )
    return expanded


def plot_case(row, output_path, max_text_chars):
    events = expand_events(row.get("route_trace", []))
    if not events:
        raise ValueError("No route trace events found for this question.")

    box_w = 1.55
    gap = 0.22
    box_h = 1.4
    left_margin = 0.5
    bottom = 0.9
    fig_w = max(12, len(events) * 1.45)
    fig_h = 5.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, left_margin + len(events) * (box_w + gap) + 0.6)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    title = (
        f"Observe-and-Rollback Case Study | question_id={row['question_id']} | "
        f"small_correct={row['small_is_correct']} | scheduled_correct={row['scheduled_is_correct']} | "
        f"handoff_count={row['handoff_count']}"
    )
    ax.text(0.02, 3.0, title, fontsize=14, fontweight="bold", ha="left", va="top")

    legend_y = 2.65
    legend_items = [
        ("small_accept", "SLM accepted chunk"),
        ("small_observe_rollback", "SLM observed then rolled back"),
        ("large_handoff", "LLM replacement chunk"),
    ]
    legend_x = 0.5
    for key, label in legend_items:
        style = EVENT_STYLES[key]
        ax.add_patch(Rectangle((legend_x, legend_y), 0.28, 0.18, facecolor=style["color"], edgecolor="black", linewidth=0.8))
        ax.text(legend_x + 0.34, legend_y + 0.09, label, fontsize=10, va="center", ha="left")
        legend_x += 2.7

    rollback_positions = []
    large_positions = []
    x = left_margin
    for idx, event in enumerate(events):
        style = EVENT_STYLES[event["event"]]
        ax.add_patch(
            Rectangle(
                (x, bottom),
                box_w,
                box_h,
                facecolor=style["color"],
                edgecolor="black",
                linewidth=1.0,
                alpha=0.95,
            )
        )
        ax.text(x + box_w / 2, bottom + box_h - 0.12, event["title"], fontsize=10.5, fontweight="bold", ha="center", va="top", color=style["text_color"])
        ax.text(x + box_w / 2, bottom + box_h - 0.33, event["subtitle"], fontsize=8.5, ha="center", va="top", color=style["text_color"])
        body = wrap_label(shorten(event["text"], max_text_chars), width=18)
        ax.text(x + 0.08, bottom + 0.12, body, fontsize=8.4, ha="left", va="bottom", color=style["text_color"])

        if event["event"] == "small_observe_rollback":
            rollback_positions.append((idx, x + box_w / 2))
        elif event["event"] == "large_handoff":
            large_positions.append((idx, x + box_w / 2))

        if idx < len(events) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.75, bottom + box_h / 2),
                xytext=(x + box_w, bottom + box_h / 2),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#666666"),
            )
        x += box_w + gap

    # Draw dashed arrows from each rollback chunk to the next large chunk.
    large_idx = 0
    for _, rollback_x in rollback_positions:
        while large_idx < len(large_positions) and large_positions[large_idx][1] < rollback_x:
            large_idx += 1
        if large_idx < len(large_positions):
            target_x = large_positions[large_idx][1]
            ax.annotate(
                "rollback -> LLM repair",
                xy=(target_x, bottom + box_h + 0.03),
                xytext=(rollback_x, bottom + box_h + 0.38),
                fontsize=9,
                ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="--", color="#333333"),
            )
            large_idx += 1

    footer = (
        f"small_answer={row['small_final_answer']} | scheduled_answer={row['scheduled_final_answer']}"
    )
    ax.text(0.5, 0.28, footer, fontsize=10.5, ha="left", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_trace_rows(args.trace_path, threshold=args.threshold)
    row = next((item for item in rows if int(item["question_id"]) == int(args.question_id)), None)
    if row is None:
        raise KeyError(f"question_id={args.question_id} not found in trace file.")

    output_path = args.output_path
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        threshold_tag = "auto" if args.threshold is None else str(args.threshold).replace(".", "p")
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"observe_rollback_case_q{args.question_id}_thr{threshold_tag}.png")

    plot_case(row=row, output_path=output_path, max_text_chars=args.max_text_chars)
    print(f"Saved case figure to: {output_path}")


if __name__ == "__main__":
    main()
