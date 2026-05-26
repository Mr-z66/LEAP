#!/usr/bin/env python
"""Merge trajectory shard .pt files into one trajectory .pt file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing trajectory shards.")
    parser.add_argument("--pattern", required=True, help="Glob pattern, e.g. livecodebench_v5_test_*_15b.pt.")
    parser.add_argument("--output-path", required=True, help="Merged output .pt path.")
    parser.add_argument("--summary-path", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def load_shard(path: Path) -> List[dict]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}, got {type(rows).__name__}")
    return rows


def question_key(value) -> str:
    return str(value if value is not None else "-1")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    shard_paths = sorted(input_dir.glob(args.pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards matched: {input_dir / args.pattern}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    merged: List[dict] = []
    summary = {
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "output_path": str(output_path),
        "shards": [],
    }
    for shard_path in shard_paths:
        rows = load_shard(shard_path)
        merged.extend(rows)
        summary["shards"].append({"path": str(shard_path), "rows": len(rows)})

    question_ids = [question_key(row.get("question_id", "-1")) for row in merged if isinstance(row, dict)]
    duplicates = len(question_ids) - len(set(question_ids))
    summary.update(
        {
            "merged_rows": len(merged),
            "merged_questions": len(set(question_ids)),
            "duplicate_question_ids": duplicates,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Wrote merged trajectories: {output_path}")


if __name__ == "__main__":
    main()
