import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare next-chunk risk labels from a strict chunk-labeled dataset."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to strict chunk-level labeled .pt dataset (label: 1=correct, 0=error).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save relabeled .pt dataset.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Number of future chunks to inspect when assigning next-risk label.",
    )
    parser.add_argument(
        "--drop-tail-without-full-horizon",
        action="store_true",
        help="Drop chunks that do not have a full future horizon available.",
    )
    return parser.parse_args()


def build_question_records(dataset):
    question_records = {}
    for item in dataset:
        label = int(item.get("label", -1))
        if label not in {0, 1}:
            continue
        question_id = int(item["question_id"])
        record = question_records.setdefault(question_id, {"chunks": []})
        record["chunks"].append(item)

    for record in question_records.values():
        record["chunks"] = sorted(record["chunks"], key=lambda chunk: int(chunk["chunk_id"]))
    return question_records


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    dataset = torch.load(input_path, weights_only=False)
    question_records = build_question_records(dataset)

    relabeled_rows = []
    positive_count = 0
    negative_count = 0
    skipped_rows = 0

    for question_id, record in sorted(question_records.items()):
        chunks = record["chunks"]
        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):
            future_chunks = chunks[index + 1 : index + 1 + args.horizon]
            if args.drop_tail_without_full_horizon and len(future_chunks) < args.horizon:
                skipped_rows += 1
                continue
            if not future_chunks:
                skipped_rows += 1
                continue

            next1_error = 1 if int(future_chunks[0]["label"]) == 0 else 0
            nextk_error = 1 if any(int(future_chunk["label"]) == 0 for future_chunk in future_chunks) else 0

            first_error_offset = None
            for offset, future_chunk in enumerate(future_chunks, start=1):
                if int(future_chunk["label"]) == 0:
                    first_error_offset = offset
                    break

            new_item = dict(chunk)
            new_item["current_label"] = int(chunk["label"])
            new_item["label"] = int(nextk_error)
            new_item["next1_error"] = int(next1_error)
            new_item["nextk_error"] = int(nextk_error)
            new_item["risk_horizon"] = int(args.horizon)
            new_item["future_window_size"] = int(len(future_chunks))
            new_item["first_future_error_offset"] = first_error_offset

            relabeled_rows.append(new_item)
            if nextk_error:
                positive_count += 1
            else:
                negative_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(relabeled_rows, output_path)

    print(f"Saved next-chunk risk dataset to: {output_path}")
    print(f"Questions kept: {len(question_records)}")
    print(f"Rows kept: {len(relabeled_rows)} | skipped tail rows: {skipped_rows}")
    print(f"Horizon: {args.horizon}")
    print(f"Label distribution: {{0: {negative_count}, 1: {positive_count}}}")


if __name__ == "__main__":
    main()
