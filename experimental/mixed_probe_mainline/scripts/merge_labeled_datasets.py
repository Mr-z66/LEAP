#!/usr/bin/env python
"""Merge multiple labeled chunk datasets into one mixed-domain training file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "dataset" / "mixed_probe_labels_fallback_second_pass"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "dataset" / "mixed_gsm8k_svamp_calib_labels.pt"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "result" / "analysis_outputs" / "mixed_gsm8k_svamp_merge_summary.json"
DEFAULT_DATASETS = "gsm8k_calib,svamp_calib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing per-split labeled .pt files.")
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset keys to merge, e.g. gsm8k_calib,svamp_calib,math500_calib.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Merged .pt output path.")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH), help="JSON summary output path.")
    return parser.parse_args()


def parse_dataset_keys(text: str) -> List[str]:
    keys = [part.strip() for part in text.split(",") if part.strip()]
    if not keys:
        raise ValueError("Expected at least one dataset key.")
    return keys


def infer_dataset_name(key: str, rows: List[dict]) -> str:
    if rows:
        meta = rows[0].get("source_meta", {})
        if isinstance(meta, dict):
            mixed_meta = meta.get("_mixed_mainline")
            if isinstance(mixed_meta, dict) and mixed_meta.get("dataset_name"):
                return str(mixed_meta["dataset_name"])
        answer_type = rows[0].get("answer_type")
        if answer_type:
            answer_type = str(answer_type)
            if answer_type.startswith("gsm8k"):
                return "gsm8k"
            if answer_type.startswith("svamp"):
                return "svamp"
            if answer_type.startswith("math500"):
                return "math500"
            if answer_type.startswith("livecodebench"):
                return "livecodebench_v5"
    return key.split("_", 1)[0]


def load_rows(path: Path) -> List[dict]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty row list in {path}")
    return rows


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def merge_rows(dataset_key: str, dataset_index: int, rows: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    stats = Counter()
    remapped: List[dict] = []
    dataset_name = infer_dataset_name(dataset_key, rows)
    question_id_map: Dict[int, int] = {}
    next_question_local = 0

    for row in rows:
        original_question_id = int(row["question_id"])
        if original_question_id not in question_id_map:
            question_id_map[original_question_id] = dataset_index * 1_000_000 + next_question_local
            next_question_local += 1

        new_question_id = question_id_map[original_question_id]
        merged = dict(row)
        merged["question_id"] = new_question_id
        merged["original_question_id"] = original_question_id
        merged["mixed_dataset_key"] = dataset_key
        merged["mixed_dataset_name"] = dataset_name
        merged["mixed_split_name"] = dataset_key.split("_", 1)[1] if "_" in dataset_key else "unknown"

        source_meta = dict(merged.get("source_meta", {}) or {})
        source_meta["_mixed_probe_merge"] = {
            "dataset_key": dataset_key,
            "dataset_name": dataset_name,
            "original_question_id": original_question_id,
            "merged_question_id": new_question_id,
        }
        merged["source_meta"] = source_meta

        remapped.append(merged)
        stats["rows"] += 1
        stats["questions"] = len(question_id_map)
        if int(merged.get("label", -1)) == 1:
            stats["label_1"] += 1
        elif int(merged.get("label", -1)) == 0:
            stats["label_0"] += 1

    return remapped, dict(stats)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    dataset_keys = parse_dataset_keys(args.datasets)

    merged_rows: List[dict] = []
    summary = {
        "input_dir": relpath(input_dir),
        "datasets": {},
        "merged_rows": 0,
        "merged_questions": 0,
    }

    for dataset_index, dataset_key in enumerate(dataset_keys, start=1):
        label_path = input_dir / f"{dataset_key}_labels.pt"
        rows = load_rows(label_path)
        remapped_rows, stats = merge_rows(dataset_key, dataset_index, rows)
        merged_rows.extend(remapped_rows)
        summary["datasets"][dataset_key] = {
            "path": relpath(label_path),
            **stats,
        }

    merged_question_ids = {int(row["question_id"]) for row in merged_rows}
    summary["merged_rows"] = len(merged_rows)
    summary["merged_questions"] = len(merged_question_ids)
    summary["label_counts"] = dict(Counter(int(row.get("label", -1)) for row in merged_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_rows, output_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Wrote merged labels: {output_path}")
    print(f"Wrote merge summary: {summary_path}")


if __name__ == "__main__":
    main()
