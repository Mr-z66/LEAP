import argparse
import json
import math
import os
import statistics
from collections import Counter

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "analysis_outputs", "latent_drift_summary.json")
DEFAULT_FEATURE_NAME = "boundary_hidden_state"
DEFAULT_METRIC = "cosine"
DEFAULT_SPIKE_STD_SCALE = 2.0
DEFAULT_LOCAL_WINDOW_BEFORE = 3
DEFAULT_LOCAL_WINDOW_AFTER = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline latent drift analysis on chunk-level strict labeled trajectories."
    )
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled chunk dataset (.pt).")
    parser.add_argument(
        "--feature-name",
        default=DEFAULT_FEATURE_NAME,
        choices=["boundary_hidden_state", "mean_hidden_state"],
        help="Hidden-state feature used to compute inter-chunk drift.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["cosine", "l2"],
        help="Drift metric between adjacent chunk representations.",
    )
    parser.add_argument(
        "--spike-std-scale",
        type=float,
        default=DEFAULT_SPIKE_STD_SCALE,
        help="Treat drift > mean + scale * std as a spike.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save a JSON summary with aggregate statistics and hard-case examples.",
    )
    parser.add_argument(
        "--top-k-hard-cases",
        type=int,
        default=15,
        help="How many high-drift/error examples to save in the output JSON.",
    )
    parser.add_argument(
        "--local-window-before",
        type=int,
        default=DEFAULT_LOCAL_WINDOW_BEFORE,
        help="How many chunks before first_error_chunk_id to inspect for local rollback anchors.",
    )
    parser.add_argument(
        "--local-window-after",
        type=int,
        default=DEFAULT_LOCAL_WINDOW_AFTER,
        help="How many chunks after first_error_chunk_id to inspect for local rollback anchors.",
    )
    return parser.parse_args()


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    return statistics.median(values) if values else 0.0


def safe_stdev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def cosine_drift(current_vec, prev_vec):
    current_norm = np.linalg.norm(current_vec)
    prev_norm = np.linalg.norm(prev_vec)
    if current_norm == 0.0 or prev_norm == 0.0:
        return 0.0
    similarity = float(np.dot(current_vec, prev_vec) / (current_norm * prev_norm))
    similarity = max(min(similarity, 1.0), -1.0)
    return 1.0 - similarity


def l2_drift(current_vec, prev_vec):
    return float(np.linalg.norm(current_vec - prev_vec))


def compute_drift(current_vec, prev_vec, metric):
    if metric == "cosine":
        return cosine_drift(current_vec, prev_vec)
    if metric == "l2":
        return l2_drift(current_vec, prev_vec)
    raise ValueError(f"Unsupported metric: {metric}")


def load_dataset(path):
    print(f"Loading strict labeled dataset from: {path}")
    dataset = torch.load(path)
    if not dataset:
        raise ValueError(f"Dataset is empty: {path}")
    return dataset


def group_by_question(dataset):
    grouped = {}
    for item in dataset:
        qid = int(item["question_id"])
        record = grouped.setdefault(
            qid,
            {
                "question_id": qid,
                "question": item.get("question", ""),
                "ground_truth_final_answer": item.get("ground_truth_final_answer"),
                "chunks": [],
            },
        )
        record["chunks"].append(item)

    for record in grouped.values():
        record["chunks"] = sorted(record["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    return grouped


def summarize_bucket(values):
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": safe_median(values),
        "std": safe_stdev(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def build_question_analysis(question_record, feature_name, metric):
    chunks = question_record["chunks"]
    drift_rows = []
    first_error_chunk_id = None
    for chunk in chunks:
        if int(chunk["label"]) == 0:
            first_error_chunk_id = int(chunk["chunk_id"])
            break

    prev_vec = None
    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        vec = tensor_to_numpy(chunk[feature_name]).reshape(-1).astype(np.float32)
        drift = None if prev_vec is None else compute_drift(vec, prev_vec, metric)
        drift_rows.append(
            {
                "question_id": question_record["question_id"],
                "chunk_id": chunk_id,
                "label": int(chunk["label"]),
                "relative_position": float(chunk_id / max(len(chunks) - 1, 1)),
                "drift": drift,
                "chunk_text": str(chunk.get("chunk_text", "")),
                "is_first_error_chunk": first_error_chunk_id is not None and chunk_id == first_error_chunk_id,
                "is_before_first_error": first_error_chunk_id is not None and chunk_id < first_error_chunk_id,
                "is_after_first_error": first_error_chunk_id is not None and chunk_id > first_error_chunk_id,
            }
        )
        prev_vec = vec

    return {
        "question_id": question_record["question_id"],
        "question": question_record["question"],
        "ground_truth_final_answer": question_record["ground_truth_final_answer"],
        "num_chunks": len(chunks),
        "first_error_chunk_id": first_error_chunk_id,
        "drift_rows": drift_rows,
    }


def aggregate_question_analyses(
    question_analyses,
    spike_std_scale,
    top_k_hard_cases,
    local_window_before,
    local_window_after,
):
    all_drifts = []
    correct_prefix_drifts = []
    error_chunk_drifts = []
    pre_error_drifts = []
    post_error_drifts = []
    first_error_transition_drifts = []
    per_question_max_drift = []
    questions_with_error = 0
    spike_examples = []
    rollback_anchor_counter = Counter()
    local_anchor_counter = Counter()
    local_peak_examples = []
    local_anchor_abs_distances = []
    local_anchor_signed_distances = []
    local_anchor_exact_hits = 0
    local_anchor_within_one = 0

    for analysis in question_analyses:
        drift_rows = [row for row in analysis["drift_rows"] if row["drift"] is not None]
        drift_values = [row["drift"] for row in drift_rows]
        all_drifts.extend(drift_values)

        if analysis["first_error_chunk_id"] is not None:
            questions_with_error += 1
            local_mean = safe_mean(drift_values)
            local_std = safe_stdev(drift_values)
            local_threshold = local_mean + spike_std_scale * local_std
            first_error_chunk_id = analysis["first_error_chunk_id"]

            candidate_rows = []
            for row in drift_rows:
                if row["label"] == 1:
                    correct_prefix_drifts.append(row["drift"])
                else:
                    error_chunk_drifts.append(row["drift"])

                if row["is_before_first_error"]:
                    pre_error_drifts.append(row["drift"])
                elif row["is_after_first_error"]:
                    post_error_drifts.append(row["drift"])

                if row["is_first_error_chunk"]:
                    first_error_transition_drifts.append(row["drift"])

                if row["drift"] >= local_threshold:
                    candidate_rows.append(row)

            window_start = max(1, first_error_chunk_id - local_window_before)
            window_end = first_error_chunk_id + local_window_after
            local_window_rows = [
                row
                for row in drift_rows
                if window_start <= row["chunk_id"] <= window_end
            ]
            if local_window_rows:
                local_anchor = max(local_window_rows, key=lambda row: row["drift"])
                signed_distance = local_anchor["chunk_id"] - first_error_chunk_id
                abs_distance = abs(signed_distance)
                local_anchor_signed_distances.append(signed_distance)
                local_anchor_abs_distances.append(abs_distance)
                local_anchor_counter[local_anchor["chunk_id"]] += 1
                if abs_distance == 0:
                    local_anchor_exact_hits += 1
                if abs_distance <= 1:
                    local_anchor_within_one += 1
                local_peak_examples.append(
                    {
                        "question_id": analysis["question_id"],
                        "first_error_chunk_id": first_error_chunk_id,
                        "local_window_start": window_start,
                        "local_window_end": window_end,
                        "suggested_local_anchor_chunk_id": local_anchor["chunk_id"],
                        "signed_distance_to_first_error": signed_distance,
                        "abs_distance_to_first_error": abs_distance,
                        "drift": local_anchor["drift"],
                        "chunk_text": local_anchor["chunk_text"],
                        "question": analysis["question"],
                    }
                )

            if candidate_rows:
                anchor = min(candidate_rows, key=lambda row: row["chunk_id"])
                rollback_anchor_counter[anchor["chunk_id"]] += 1
                spike_examples.append(
                    {
                        "question_id": analysis["question_id"],
                        "first_error_chunk_id": analysis["first_error_chunk_id"],
                        "suggested_rollback_chunk_id": anchor["chunk_id"],
                        "drift": anchor["drift"],
                        "local_spike_threshold": local_threshold,
                        "chunk_text": anchor["chunk_text"],
                        "question": analysis["question"],
                    }
                )

        if drift_values:
            per_question_max_drift.append(
                {
                    "question_id": analysis["question_id"],
                    "max_drift": max(drift_values),
                    "first_error_chunk_id": analysis["first_error_chunk_id"],
                    "question": analysis["question"],
                }
            )

    per_question_max_drift.sort(key=lambda item: item["max_drift"], reverse=True)
    spike_examples.sort(key=lambda item: item["drift"], reverse=True)

    global_mean = safe_mean(all_drifts)
    global_std = safe_stdev(all_drifts)
    global_spike_threshold = global_mean + spike_std_scale * global_std

    global_spike_count = sum(1 for value in all_drifts if value >= global_spike_threshold)
    first_error_spike_count = sum(1 for value in first_error_transition_drifts if value >= global_spike_threshold)
    local_peak_examples.sort(key=lambda item: (item["abs_distance_to_first_error"], -item["drift"]))

    return {
        "num_questions": len(question_analyses),
        "questions_with_error": questions_with_error,
        "drift_metric": {
            "global": summarize_bucket(all_drifts),
            "correct_prefix": summarize_bucket(correct_prefix_drifts),
            "error_chunk": summarize_bucket(error_chunk_drifts),
            "pre_error": summarize_bucket(pre_error_drifts),
            "post_error": summarize_bucket(post_error_drifts),
            "first_error_transition": summarize_bucket(first_error_transition_drifts),
        },
        "spike_analysis": {
            "global_spike_threshold": global_spike_threshold,
            "global_spike_count": global_spike_count,
            "first_error_spike_count": first_error_spike_count,
            "first_error_spike_rate": first_error_spike_count / max(len(first_error_transition_drifts), 1),
            "rollback_anchor_counter": dict(rollback_anchor_counter.most_common()),
        },
        "local_window_analysis": {
            "window_before": local_window_before,
            "window_after": local_window_after,
            "count": len(local_anchor_abs_distances),
            "anchor_abs_distance": summarize_bucket(local_anchor_abs_distances),
            "anchor_signed_distance": summarize_bucket(local_anchor_signed_distances),
            "exact_hit_rate": local_anchor_exact_hits / max(len(local_anchor_abs_distances), 1),
            "within_one_chunk_rate": local_anchor_within_one / max(len(local_anchor_abs_distances), 1),
            "local_anchor_counter": dict(local_anchor_counter.most_common()),
        },
        "top_max_drift_questions": per_question_max_drift[:top_k_hard_cases],
        "top_spike_examples": spike_examples[:top_k_hard_cases],
        "top_local_peak_examples": local_peak_examples[:top_k_hard_cases],
    }


def print_summary(summary, feature_name, metric, spike_std_scale):
    print("\nLatent drift summary")
    print("=" * 50)
    print(f"Feature: {feature_name}")
    print(f"Metric: {metric}")
    print(f"Questions: {summary['num_questions']}")
    print(f"Questions with error: {summary['questions_with_error']}")

    print("\nDrift distribution")
    print("=" * 50)
    for bucket_name, bucket_stats in summary["drift_metric"].items():
        print(
            f"{bucket_name}: count={bucket_stats['count']} "
            f"mean={bucket_stats['mean']:.6f} median={bucket_stats['median']:.6f} "
            f"std={bucket_stats['std']:.6f}"
        )

    spike = summary["spike_analysis"]
    print("\nSpike analysis")
    print("=" * 50)
    print(f"Spike threshold (mean + {spike_std_scale:.2f} * std): {spike['global_spike_threshold']:.6f}")
    print(f"Global spike count: {spike['global_spike_count']}")
    print(f"First-error transitions above global spike threshold: {spike['first_error_spike_count']}")
    print(f"First-error spike rate: {spike['first_error_spike_rate']:.4f}")

    local_window = summary["local_window_analysis"]
    print("\nLocal window analysis")
    print("=" * 50)
    print(
        f"Window: first_error - {local_window['window_before']} to + {local_window['window_after']}"
    )
    print(f"Questions analyzed in local window: {local_window['count']}")
    print(
        f"Anchor abs distance mean={local_window['anchor_abs_distance']['mean']:.4f} "
        f"median={local_window['anchor_abs_distance']['median']:.4f}"
    )
    print(f"Exact-hit rate: {local_window['exact_hit_rate']:.4f}")
    print(f"Within-one-chunk rate: {local_window['within_one_chunk_rate']:.4f}")

    print("\nTop high-drift questions")
    print("=" * 50)
    for item in summary["top_max_drift_questions"][:10]:
        print(
            f"qid={item['question_id']} | max_drift={item['max_drift']:.6f} "
            f"| first_error_chunk_id={item['first_error_chunk_id']}"
        )

    print("\nTop rollback anchor candidates")
    print("=" * 50)
    for item in summary["top_spike_examples"][:10]:
        print(
            f"qid={item['question_id']} | first_error={item['first_error_chunk_id']} "
            f"| suggested_anchor={item['suggested_rollback_chunk_id']} | drift={item['drift']:.6f}"
        )

    print("\nTop local-window anchor examples")
    print("=" * 50)
    for item in summary["top_local_peak_examples"][:10]:
        print(
            f"qid={item['question_id']} | first_error={item['first_error_chunk_id']} "
            f"| local_anchor={item['suggested_local_anchor_chunk_id']} "
            f"| abs_dist={item['abs_distance_to_first_error']} | drift={item['drift']:.6f}"
        )


def main():
    args = parse_args()
    dataset = load_dataset(args.label_path)
    grouped = group_by_question(dataset)
    question_analyses = [
        build_question_analysis(record, args.feature_name, args.metric)
        for _, record in sorted(grouped.items(), key=lambda pair: pair[0])
    ]
    summary = aggregate_question_analyses(
        question_analyses=question_analyses,
        spike_std_scale=args.spike_std_scale,
        top_k_hard_cases=args.top_k_hard_cases,
        local_window_before=args.local_window_before,
        local_window_after=args.local_window_after,
    )
    print_summary(summary, args.feature_name, args.metric, args.spike_std_scale)

    output_payload = {
        "label_path": args.label_path,
        "feature_name": args.feature_name,
        "metric": args.metric,
        "spike_std_scale": args.spike_std_scale,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved latent drift analysis to: {args.output_path}")


if __name__ == "__main__":
    main()
