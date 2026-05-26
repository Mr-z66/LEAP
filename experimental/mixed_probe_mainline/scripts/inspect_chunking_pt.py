#!/usr/bin/env python
"""
Inspect chunking health for mixed-probe artifacts.

Supports two input formats:
1) Trajectories from `core_package.pipelines.build_dataset`:
   List[{"question_id": ..., "chunks": [ ...chunk dicts... ], "chunking_method": ..., "chunking_config": ...}, ...]
2) Labeled chunk rows from `core_package.pipelines.label_existing_trajectories` /
   second-pass refinements:
   List[{"question_id": ..., "chunk_id": ..., "token_count": ..., "cut_reason": ..., "label": ...}, ...]

Typical usage (on your Linux/conda env with torch):
  python experimental/mixed_probe_mainline/scripts/inspect_chunking_pt.py \
    dataset/mixed_probe_trajectories_fallback/math500_test_300_15b.pt \
    dataset/mixed_probe_labels_fallback_second_pass/math500_test_labels.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="One or more .pt files to inspect.")
    p.add_argument("--max-questions", type=int, default=None, help="Limit to first N question_ids (after sorting).")
    p.add_argument("--json", action="store_true", help="Print JSON summary (in addition to a short text summary).")
    return p.parse_args()


def is_trajectory_rows(rows: Any) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    head = rows[0]
    return isinstance(head, dict) and "chunks" in head and isinstance(head.get("chunks"), list)


def iter_chunks_from_trajectories(rows: List[dict]) -> Iterable[dict]:
    for sample in rows:
        for chunk in sample.get("chunks", []) or []:
            yield {
                "question_id": int(sample.get("question_id", -1)),
                "token_count": int(chunk.get("token_count", len(chunk.get("token_ids", []) or []))),
                "cut_reason": str(chunk.get("cut_reason", "")),
                "ambiguous_chunk": bool(chunk.get("ambiguous_chunk", False)),
                "chunking_method": sample.get("chunking_method", None),
                "chunking_config": sample.get("chunking_config", None),
            }


def iter_chunks_from_labeled_rows(rows: List[dict]) -> Iterable[dict]:
    for row in rows:
        yield {
            "question_id": int(row.get("question_id", -1)),
            "token_count": int(row.get("token_count", 0)),
            "cut_reason": str(row.get("cut_reason", "")),
            "ambiguous_chunk": bool(row.get("ambiguous_chunk", False)),
            "chunking_method": row.get("chunking_method", None),
            "chunking_config": row.get("chunking_config", None),
            "label": int(row["label"]) if "label" in row else None,
        }


def summarize_numeric(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.mean(values_sorted)),
        "median": float(statistics.median(values_sorted)),
        "min": int(values_sorted[0]),
        "max": int(values_sorted[-1]),
        "p90": float(statistics.quantiles(values_sorted, n=10)[8]) if len(values_sorted) >= 10 else float(values_sorted[-1]),
    }


def shorten_config(config: Any) -> str:
    if config is None:
        return ""
    if isinstance(config, str):
        return config
    try:
        return json.dumps(config, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(config)


def inspect_one(path: Path, max_questions: int | None) -> Dict[str, Any]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}, got {type(rows).__name__}")
    if not rows:
        return {"path": str(path), "format": "empty", "rows": 0}

    if is_trajectory_rows(rows):
        fmt = "trajectories"
        chunk_iter = list(iter_chunks_from_trajectories(rows))
        question_ids = sorted({int(r.get("question_id", -1)) for r in rows if isinstance(r, dict)})
        chunking_methods = Counter(str(r.get("chunking_method", "")) for r in rows if isinstance(r, dict))
        chunking_configs = Counter(shorten_config(r.get("chunking_config")) for r in rows if isinstance(r, dict))
    else:
        fmt = "labeled_rows"
        chunk_iter = list(iter_chunks_from_labeled_rows(rows))
        question_ids = sorted({int(r.get("question_id", -1)) for r in rows if isinstance(r, dict)})
        chunking_methods = Counter(str(r.get("chunking_method", "")) for r in rows if isinstance(r, dict))
        chunking_configs = Counter(shorten_config(r.get("chunking_config")) for r in rows if isinstance(r, dict))

    if max_questions is not None:
        keep = set(question_ids[: max_questions])
        chunk_iter = [c for c in chunk_iter if c["question_id"] in keep]
        question_ids = sorted(keep)

    per_q = defaultdict(int)
    for c in chunk_iter:
        per_q[c["question_id"]] += 1

    token_counts = [int(c.get("token_count", 0)) for c in chunk_iter if int(c.get("token_count", 0)) > 0]
    cut_reasons = Counter(str(c.get("cut_reason", "")) for c in chunk_iter)
    ambiguous = sum(1 for c in chunk_iter if c.get("ambiguous_chunk"))
    labels = Counter(int(c["label"]) for c in chunk_iter if c.get("label") is not None)

    return {
        "path": str(path),
        "format": fmt,
        "rows": len(rows),
        "questions": len(question_ids),
        "chunks": len(chunk_iter),
        "chunks_per_question": summarize_numeric(list(per_q.values())),
        "token_count": summarize_numeric(token_counts),
        "cut_reasons": dict(cut_reasons),
        "ambiguous_chunks": int(ambiguous),
        "ambiguous_rate": float(ambiguous / max(1, len(chunk_iter))),
        "labels": dict(labels) if labels else None,
        "chunking_methods": dict(chunking_methods),
        "chunking_configs_top": dict(chunking_configs.most_common(3)),
    }


def main() -> None:
    args = parse_args()
    summaries: List[Dict[str, Any]] = []
    for raw in args.paths:
        path = Path(raw)
        summary = inspect_one(path, args.max_questions)
        summaries.append(summary)

        # Short, human-readable printout
        print("\n" + str(path))
        print(f"format: {summary.get('format')} rows={summary.get('rows')} questions={summary.get('questions')} chunks={summary.get('chunks')}")
        print(f"chunks/q: {summary.get('chunks_per_question')}")
        print(f"token_count: {summary.get('token_count')}")
        print(f"ambiguous: {summary.get('ambiguous_chunks')} rate={summary.get('ambiguous_rate'):.4f}")
        if summary.get("labels"):
            print(f"labels: {summary['labels']}")
        print(f"methods: {summary.get('chunking_methods')}")
        print(f"cut_reasons_top: {dict(Counter(summary.get('cut_reasons') or {}).most_common(8))}")

    if args.json:
        print("\nJSON_SUMMARY=" + json.dumps(summaries, ensure_ascii=True))


if __name__ == "__main__":
    main()

