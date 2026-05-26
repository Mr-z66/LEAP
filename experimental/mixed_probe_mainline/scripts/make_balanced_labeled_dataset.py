#!/usr/bin/env python
"""Create a balanced labeled chunk dataset for probe training diagnostics."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Input merged labeled .pt file.")
    parser.add_argument("--output-path", required=True, help="Output balanced labeled .pt file.")
    parser.add_argument("--summary-path", default=None, help="Optional JSON summary path.")
    parser.add_argument("--neg-to-pos-ratio", type=float, default=3.0, help="Keep up to this many label=1 chunks per label=0 chunk.")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--by-domain", action="store_true", help="Sample label=1 chunks separately within each mixed_dataset_key.")
    parser.add_argument("--drop-unlabeled", action="store_true", help="Drop labels outside {0,1}.")
    return parser.parse_args()


def domain_key(row: Dict[str, Any]) -> str:
    return str(row.get("mixed_dataset_key") or row.get("mixed_dataset_name") or "unknown")


def summarize(rows: List[dict]) -> Dict[str, Any]:
    label_counts = Counter(int(row.get("label", -1)) for row in rows if "label" in row)
    domain_counts = defaultdict(Counter)
    question_ids = set()
    for row in rows:
        label = int(row.get("label", -1))
        domain_counts[domain_key(row)][label] += 1
        question_ids.add(str(row.get("question_id", "-1")))
    return {
        "rows": len(rows),
        "questions": len(question_ids),
        "label_counts": dict(label_counts),
        "domain_label_counts": {key: dict(value) for key, value in sorted(domain_counts.items())},
    }


def sample_group(rows: List[dict], ratio: float, rng: random.Random) -> List[dict]:
    label_0 = [row for row in rows if int(row.get("label", -1)) == 0]
    label_1 = [row for row in rows if int(row.get("label", -1)) == 1]
    unlabeled = [row for row in rows if int(row.get("label", -1)) not in {0, 1}]

    keep_1 = min(len(label_1), int(round(len(label_0) * ratio)))
    sampled_1 = rng.sample(label_1, keep_1) if keep_1 < len(label_1) else label_1
    return label_0 + sampled_1 + unlabeled


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = torch.load(args.input_path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {args.input_path}, got {type(rows).__name__}")

    valid_rows = [row for row in rows if int(row.get("label", -1)) in {0, 1}]
    unlabeled_rows = [] if args.drop_unlabeled else [row for row in rows if int(row.get("label", -1)) not in {0, 1}]

    if args.by_domain:
        grouped = defaultdict(list)
        for row in valid_rows:
            grouped[domain_key(row)].append(row)
        balanced = []
        for _, group_rows in sorted(grouped.items()):
            balanced.extend(sample_group(group_rows, args.neg_to_pos_ratio, rng))
        balanced.extend(unlabeled_rows)
    else:
        balanced = sample_group(valid_rows, args.neg_to_pos_ratio, rng)
        balanced.extend(unlabeled_rows)

    rng.shuffle(balanced)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(balanced, output_path)

    summary = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "neg_to_pos_ratio": args.neg_to_pos_ratio,
        "seed": args.seed,
        "by_domain": args.by_domain,
        "drop_unlabeled": args.drop_unlabeled,
        "input": summarize(rows),
        "output": summarize(balanced),
    }
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Wrote balanced labels: {output_path}")


if __name__ == "__main__":
    main()
