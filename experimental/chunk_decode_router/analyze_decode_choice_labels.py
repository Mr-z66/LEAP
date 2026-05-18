import argparse
import json
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize decode-choice labels and candidate distribution.")
    parser.add_argument("--input-path", required=True, help="Path to labeled or unlabeled decode-choice jsonl.")
    parser.add_argument("--show-examples", type=int, default=0, help="Print up to N LLM-labeled examples.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No rows found.")
        return

    per_question = defaultdict(int)
    hard = Counter()
    util = Counter()
    llm_examples = []

    for row in rows:
        qid = row.get("question_id")
        per_question[qid] += 1
        meta = row.get("label_metadata", {})
        hard[meta.get("label")] += 1
        util[meta.get("utility_label")] += 1
        if meta.get("label") == "LLM" and len(llm_examples) < args.show_examples:
            llm_examples.append(
                {
                    "question_id": qid,
                    "candidate_chunk_id": row.get("candidate_chunk_id"),
                    "relative_position": row.get("relative_position"),
                    "small_final_answer": row.get("small_final_answer"),
                    "comparison": row.get("comparison", {}),
                }
            )

    counts = list(per_question.values())
    print(f"rows={len(rows)} questions={len(per_question)}")
    print(
        "candidate_count_per_question | "
        f"min={min(counts)} max={max(counts)} avg={sum(counts) / max(len(counts), 1):.2f}"
    )
    print(f"hard_labels={dict(hard)}")
    print(f"utility_labels={dict(util)}")

    if llm_examples:
        print("sample_llm_examples=")
        for item in llm_examples:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
