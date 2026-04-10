import argparse
import collections
import os
import statistics

import torch

# ================= Default Configuration =================
DEFAULT_RELAXED_PATH = os.path.join(os.getcwd(), "gsm8k_labeled_training_data.pt")
DEFAULT_STRICT_PATH = os.path.join(os.getcwd(), "dataset", "gsm8k_labeled_training_data_strict.pt")
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one or two labeled chunk datasets.")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_RELAXED_PATH,
        help="Primary labeled dataset path. Defaults to the relaxed-label file.",
    )
    parser.add_argument(
        "--compare-path",
        default=None,
        help="Optional second labeled dataset path for side-by-side comparison.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Shortcut for analyzing the default strict-label file as the primary dataset.",
    )
    parser.add_argument(
        "--compare-default-pair",
        action="store_true",
        help="Compare the default relaxed and strict labeled files.",
    )
    return parser.parse_args()


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    return statistics.median(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0


def safe_max(values):
    return max(values) if values else 0


def load_dataset(path):
    print(f"Loading labeled chunk dataset from: {path}")
    dataset = torch.load(path)
    if not dataset:
        raise ValueError(f"The labeled dataset is empty: {path}")
    return dataset


def summarize_dataset(dataset, dataset_name):
    label_counter = collections.Counter()
    cut_reason_counter = collections.Counter()
    error_type_counter = collections.Counter()
    parse_status_counter = collections.Counter()
    chunk_counts_per_question = collections.Counter()
    first_error_chunk_positions = []
    first_error_relative_positions = []
    judge_confidences = []

    question_to_chunks = collections.defaultdict(list)
    for sample in dataset:
        question_to_chunks[int(sample["question_id"])].append(sample)

    for question_id, chunks in question_to_chunks.items():
        chunks = sorted(chunks, key=lambda item: int(item["chunk_id"]))
        chunk_counts_per_question[question_id] = len(chunks)

        first_error_index = None
        for chunk in chunks:
            label_counter[int(chunk["label"])] += 1
            cut_reason_counter[chunk.get("cut_reason", "unknown")] += 1
            error_type_counter[chunk.get("judge_error_type", "unknown")] += 1
            parse_status_counter[chunk.get("judge_parse_status", "unknown")] += 1
            judge_confidences.append(float(chunk.get("judge_confidence", 0.5)))

            if int(chunk["label"]) == 0 and first_error_index is None:
                first_error_index = int(chunk["chunk_id"])

        if first_error_index is not None:
            first_error_chunk_positions.append(first_error_index)
            first_error_relative_positions.append(first_error_index / len(chunks))

    num_questions = len(question_to_chunks)
    num_chunks = len(dataset)
    positive_chunks = label_counter.get(1, 0)
    negative_chunks = label_counter.get(0, 0)
    chunk_count_values = list(chunk_counts_per_question.values())

    return {
        "dataset_name": dataset_name,
        "num_questions": num_questions,
        "num_chunks": num_chunks,
        "positive_chunks": positive_chunks,
        "negative_chunks": negative_chunks,
        "error_ratio": negative_chunks / max(1, num_chunks),
        "avg_chunks_per_question": safe_mean(chunk_count_values),
        "median_chunks_per_question": safe_median(chunk_count_values),
        "min_chunks_per_question": safe_min(chunk_count_values),
        "max_chunks_per_question": safe_max(chunk_count_values),
        "cut_reason_counter": cut_reason_counter,
        "parse_status_counter": parse_status_counter,
        "judge_confidence_mean": safe_mean(judge_confidences),
        "judge_confidence_median": safe_median(judge_confidences),
        "error_type_counter": error_type_counter,
        "questions_with_error": len(first_error_chunk_positions),
        "avg_first_error_chunk_id": safe_mean(first_error_chunk_positions),
        "median_first_error_chunk_id": safe_median(first_error_chunk_positions),
        "avg_first_error_relative_position": safe_mean(first_error_relative_positions),
        "median_first_error_relative_position": safe_median(first_error_relative_positions),
    }


def print_counter(counter):
    for key, count in counter.most_common():
        print(f"{key}: {count}")


def print_summary(summary):
    print(f"\nDataset summary: {summary['dataset_name']}")
    print("=" * 50)
    print(f"Questions: {summary['num_questions']}")
    print(f"Chunks: {summary['num_chunks']}")
    print(f"Prefix-correct chunks (label=1): {summary['positive_chunks']}")
    print(f"Prefix-error chunks (label=0): {summary['negative_chunks']}")
    print(f"Error ratio: {summary['error_ratio']:.4f}")

    print("\nChunking distribution")
    print("=" * 50)
    print(f"Avg chunks/question: {summary['avg_chunks_per_question']:.2f}")
    print(f"Median chunks/question: {summary['median_chunks_per_question']:.2f}")
    print(f"Min chunks/question: {summary['min_chunks_per_question']}")
    print(f"Max chunks/question: {summary['max_chunks_per_question']}")
    print_counter(summary["cut_reason_counter"])

    print("\nJudge output quality")
    print("=" * 50)
    print_counter(summary["parse_status_counter"])
    print(f"Avg judge confidence: {summary['judge_confidence_mean']:.4f}")
    print(f"Median judge confidence: {summary['judge_confidence_median']:.4f}")

    print("\nError type distribution")
    print("=" * 50)
    print_counter(summary["error_type_counter"])

    print("\nFirst error position")
    print("=" * 50)
    print(f"Questions with at least one error chunk: {summary['questions_with_error']}")
    if summary["questions_with_error"] > 0:
        print(f"Avg first-error chunk id: {summary['avg_first_error_chunk_id']:.2f}")
        print(f"Median first-error chunk id: {summary['median_first_error_chunk_id']:.2f}")
        print(f"Avg first-error relative position: {summary['avg_first_error_relative_position']:.4f}")
        print(f"Median first-error relative position: {summary['median_first_error_relative_position']:.4f}")


def print_comparison(primary, secondary):
    print("\nComparison summary")
    print("=" * 50)
    print(f"Primary dataset: {primary['dataset_name']}")
    print(f"Secondary dataset: {secondary['dataset_name']}")
    print(f"Questions: {primary['num_questions']} vs {secondary['num_questions']}")
    print(f"Chunks: {primary['num_chunks']} vs {secondary['num_chunks']}")
    print(
        f"Error ratio: {primary['error_ratio']:.4f} vs {secondary['error_ratio']:.4f} "
        f"(delta {secondary['error_ratio'] - primary['error_ratio']:+.4f})"
    )
    print(
        f"Questions with error: {primary['questions_with_error']} vs {secondary['questions_with_error']} "
        f"(delta {secondary['questions_with_error'] - primary['questions_with_error']:+d})"
    )
    print(
        f"Avg first-error chunk id: {primary['avg_first_error_chunk_id']:.2f} vs "
        f"{secondary['avg_first_error_chunk_id']:.2f}"
    )
    print(
        f"Avg first-error relative position: {primary['avg_first_error_relative_position']:.4f} vs "
        f"{secondary['avg_first_error_relative_position']:.4f}"
    )
    print(
        f"Avg judge confidence: {primary['judge_confidence_mean']:.4f} vs "
        f"{secondary['judge_confidence_mean']:.4f}"
    )

    print("\nCut reason comparison")
    print("=" * 50)
    all_cut_reasons = sorted(set(primary["cut_reason_counter"]) | set(secondary["cut_reason_counter"]))
    for reason in all_cut_reasons:
        print(
            f"{reason}: {primary['cut_reason_counter'].get(reason, 0)} vs "
            f"{secondary['cut_reason_counter'].get(reason, 0)}"
        )

    print("\nError type comparison")
    print("=" * 50)
    all_error_types = sorted(set(primary["error_type_counter"]) | set(secondary["error_type_counter"]))
    for error_type in all_error_types:
        print(
            f"{error_type}: {primary['error_type_counter'].get(error_type, 0)} vs "
            f"{secondary['error_type_counter'].get(error_type, 0)}"
        )


args = parse_args()

primary_path = DEFAULT_STRICT_PATH if args.strict else args.data_path
compare_path = args.compare_path

if args.compare_default_pair:
    primary_path = DEFAULT_RELAXED_PATH
    compare_path = DEFAULT_STRICT_PATH

primary_dataset = load_dataset(primary_path)
primary_summary = summarize_dataset(primary_dataset, os.path.basename(primary_path))
print_summary(primary_summary)

if compare_path is not None:
    secondary_dataset = load_dataset(compare_path)
    secondary_summary = summarize_dataset(secondary_dataset, os.path.basename(compare_path))
    print_summary(secondary_summary)
    print_comparison(primary_summary, secondary_summary)
