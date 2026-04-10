import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Count labeled questions and chunks in a strict-label output file.")
    parser.add_argument(
        "--output-path",
        default="dataset/gsm8k_labeled_training_data_strict.pt",
        help="Path to the labeled strict dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
        print("questions=0 chunks=0")
        return

    dataset = torch.load(args.output_path)
    question_ids = {int(item["question_id"]) for item in dataset}
    print(f"questions={len(question_ids)} chunks={len(dataset)}")


if __name__ == "__main__":
    main()
