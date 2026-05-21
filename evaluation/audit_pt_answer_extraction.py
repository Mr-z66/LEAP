import argparse
import json
import os
import re
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core_package.answer_registry import check_answer_correctness, get_answer_extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit answer extraction quality for chunked trajectory .pt files."
    )
    parser.add_argument("--pt-path", required=True, help="Path to a build_dataset trajectory .pt file.")
    parser.add_argument("--answer-type", required=True, help="Answer extractor/correctness protocol.")
    parser.add_argument("--max-print", type=int, default=30, help="Max suspicious examples to print.")
    parser.add_argument("--output-jsonl", default=None, help="Optional path to write suspicious examples.")
    return parser.parse_args()


def numeric_pattern(value):
    text = str(value).strip().replace(",", "")
    escaped = re.escape(text)
    if re.fullmatch(r"-?\d+", text):
        return re.compile(rf"(?<![\d.\-]){escaped}(?:\.0+)?(?![\d.])")
    return re.compile(rf"(?<![\d.\-]){escaped}(?![\d.])")


def snippet_around(text, match):
    start = max(0, match.start() - 140)
    end = min(len(text), match.end() + 180)
    return text[start:end].replace("\n", " ")


def main():
    args = parse_args()
    data = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    extractor = get_answer_extractor(args.answer_type)

    stored_mismatches = []
    wrong_gold_appears = []
    missing_extractions = []
    missing_boxed = []
    recomputed_correct = 0
    stored_correct = 0

    for item in data:
        question_id = item.get("question_id")
        generated_text = str(item.get("generated_text", "") or "")
        gold = item.get("ground_truth_final_answer")
        stored_answer = item.get("model_final_answer")
        stored_is_correct = bool(item.get("is_final_correct", False))
        stored_correct += int(stored_is_correct)

        extracted_answer, has_answer = extractor(generated_text)
        recomputed_is_correct = has_answer and check_answer_correctness(
            extracted_answer,
            gold,
            args.answer_type,
        )
        recomputed_correct += int(recomputed_is_correct)

        if not has_answer:
            missing_extractions.append(
                {
                    "question_id": question_id,
                    "ground_truth_final_answer": gold,
                    "generated_tail": generated_text[-500:].replace("\n", " "),
                }
            )

        if "boxed" in args.answer_type and "\\boxed{" not in generated_text:
            missing_boxed.append(
                {
                    "question_id": question_id,
                    "ground_truth_final_answer": gold,
                    "extracted_answer": extracted_answer,
                    "generated_tail": generated_text[-500:].replace("\n", " "),
                }
            )

        if str(stored_answer) != str(extracted_answer) or stored_is_correct != bool(recomputed_is_correct):
            stored_mismatches.append(
                {
                    "question_id": question_id,
                    "stored_answer": stored_answer,
                    "recomputed_answer": extracted_answer,
                    "stored_is_correct": stored_is_correct,
                    "recomputed_is_correct": bool(recomputed_is_correct),
                    "ground_truth_final_answer": gold,
                    "has_extracted_answer": has_answer,
                }
            )

        if not recomputed_is_correct and gold is not None:
            match = numeric_pattern(gold).search(generated_text.replace(",", ""))
            if match:
                wrong_gold_appears.append(
                    {
                        "question_id": question_id,
                        "recomputed_answer": extracted_answer,
                        "ground_truth_final_answer": gold,
                        "has_extracted_answer": has_answer,
                        "snippet": snippet_around(generated_text.replace(",", ""), match),
                    }
                )

    total = len(data)
    print(f"pt_path={args.pt_path}")
    print(f"answer_type={args.answer_type}")
    print(f"rows={total}")
    print(f"stored_accuracy={stored_correct / total:.4f}")
    print(f"recomputed_accuracy={recomputed_correct / total:.4f}")
    print(f"stored_vs_recomputed_mismatches={len(stored_mismatches)}")
    print(f"missing_extractions={len(missing_extractions)}")
    print(f"missing_boxed_outputs={len(missing_boxed)}")
    print(f"wrong_but_gold_appears_in_generation={len(wrong_gold_appears)}")

    suspicious = []
    for bucket_name, rows in [
        ("stored_vs_recomputed_mismatch", stored_mismatches),
        ("missing_extraction", missing_extractions),
        ("missing_boxed_output", missing_boxed),
        ("wrong_but_gold_appears", wrong_gold_appears),
    ]:
        for row in rows:
            suspicious.append({"bucket": bucket_name, **row})

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as handle:
            for row in suspicious:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote_suspicious_jsonl={args.output_jsonl}")

    for row in suspicious[: args.max_print]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
