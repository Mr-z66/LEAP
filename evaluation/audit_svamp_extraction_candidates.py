import argparse
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find SVAMP rows judged wrong even though the gold number appears in the generated reasoning."
    )
    parser.add_argument("--result-path", required=True, help="JSON output from evaluation.evaluate_model_only_accuracy.")
    parser.add_argument("--max-print", type=int, default=30)
    return parser.parse_args()


def numeric_pattern(value):
    text = str(value).strip().replace(",", "")
    escaped = re.escape(text)
    if re.fullmatch(r"-?\d+", text):
        return re.compile(rf"(?<![\d.\-]){escaped}(?:\.0+)?(?![\d.])")
    return re.compile(rf"(?<![\d.\-]){escaped}(?![\d.])")


def main():
    args = parse_args()
    with open(args.result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("rows", [])
    wrong_rows = [row for row in rows if not bool(row.get("is_correct", False))]
    candidates = []
    for row in wrong_rows:
        gold = row.get("ground_truth_final_answer")
        reasoning = str(row.get("reasoning", ""))
        if gold is None:
            continue
        match = numeric_pattern(gold).search(reasoning.replace(",", ""))
        if not match:
            continue
        start = max(0, match.start() - 100)
        end = min(len(reasoning), match.end() + 140)
        candidates.append(
            {
                "question_id": row.get("question_id"),
                "pred_final_answer": row.get("pred_final_answer"),
                "ground_truth_final_answer": gold,
                "has_extracted_answer": row.get("has_extracted_answer"),
                "snippet": reasoning[start:end].replace("\n", " "),
            }
        )

    total = int(payload.get("questions_total") or len(rows))
    print(f"rows={len(rows)} total={total}")
    print(f"wrong_rows={len(wrong_rows)}")
    print(f"wrong_but_gold_appears_in_reasoning={len(candidates)}")
    for item in candidates[: args.max_print]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
