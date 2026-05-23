#!/usr/bin/env python
"""Prepare fixed calibration/test splits for the mixed-domain LEAP mainline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "experimental" / "mixed_probe_mainline" / "configs" / "splits.default.json"


QUESTION_FIELDS = (
    "question",
    "problem",
    "question_content",
    "question_concat",
    "Question",
    "Body",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to split config JSON.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing/empty source files instead of failing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    return parser.parse_args()


def repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], overwrite: bool) -> int:
    rows = list(rows)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def question_text(row: Dict[str, Any]) -> str:
    if row.get("question_concat"):
        return str(row["question_concat"])
    if row.get("Body") is not None or row.get("Question") is not None:
        return f"{row.get('Body', '')} {row.get('Question', '')}"
    for field in QUESTION_FIELDS:
        if row.get(field) is not None:
            return str(row[field])
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def fingerprint(row: Dict[str, Any]) -> str:
    text = " ".join(question_text(row).strip().lower().split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sample_rows(rows: List[Dict[str, Any]], size: Optional[int], rng: random.Random) -> List[Dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if size is None:
        return shuffled
    return shuffled[: min(size, len(shuffled))]


def split_one_source(
    rows: List[Dict[str, Any]],
    calib_size: int,
    test_size: Optional[int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    calib = shuffled[: min(calib_size, len(shuffled))]
    remaining = shuffled[len(calib):]
    test = remaining if test_size is None else remaining[: min(test_size, len(remaining))]
    return calib, test


def load_required(path: Path, allow_missing: bool) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        message = f"missing source: {path}"
        if allow_missing:
            return [], message
        raise FileNotFoundError(message)
    if path.stat().st_size == 0:
        message = f"empty source: {path}"
        if allow_missing:
            return [], message
        raise ValueError(message)
    return load_jsonl(path), None


def source_summary(path: Path, rows: List[Dict[str, Any]], warning: Optional[str]) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "rows": len(rows),
        "warning": warning,
    }


def add_split_metadata(rows: List[Dict[str, Any]], dataset_name: str, split_name: str) -> List[Dict[str, Any]]:
    output = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("question_id", row.get("question_id", row.get("id", row.get("ID", row.get("unique_id", idx)))))
        item["_mixed_mainline"] = {
            "dataset_name": dataset_name,
            "split": split_name,
            "split_index": idx,
            "question_fingerprint": fingerprint(row),
        }
        output.append(item)
    return output


def overlap_count(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> int:
    return len({fingerprint(row) for row in left} & {fingerprint(row) for row in right})


def prepare(config: Dict[str, Any], output_dir: Path, seed: int, allow_missing: bool, overwrite: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "seed": seed,
        "output_dir": str(output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir),
        "datasets": {},
        "overlaps": {},
    }
    split_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for dataset_name, spec in config["datasets"].items():
        dataset_audit: Dict[str, Any] = {"sources": {}}
        if "source" in spec:
            source_path = repo_path(spec["source"])
            rows, warning = load_required(source_path, allow_missing)
            dataset_audit["sources"]["source"] = source_summary(source_path, rows, warning)
            if rows:
                calib, test = split_one_source(rows, int(spec["calib_size"]), spec.get("test_size"), rng)
            else:
                calib, test = [], []
        else:
            calib_path = repo_path(spec["calib_source"])
            test_path = repo_path(spec["test_source"])
            calib_source, calib_warning = load_required(calib_path, allow_missing)
            test_source, test_warning = load_required(test_path, allow_missing)
            dataset_audit["sources"]["calib_source"] = source_summary(calib_path, calib_source, calib_warning)
            dataset_audit["sources"]["test_source"] = source_summary(test_path, test_source, test_warning)
            calib = sample_rows(calib_source, spec.get("calib_size"), rng) if calib_source else []
            test = sample_rows(test_source, spec.get("test_size"), rng) if test_source else []

        calib = add_split_metadata(calib, dataset_name, "calib")
        test = add_split_metadata(test, dataset_name, "test")
        split_rows[dataset_name] = {"calib": calib, "test": test}

        calib_path_out = output_dir / f"{dataset_name}_calib.jsonl"
        test_path_out = output_dir / f"{dataset_name}_test.jsonl"
        dataset_audit["outputs"] = {
            "calib": {
                "path": str(calib_path_out.relative_to(REPO_ROOT) if calib_path_out.is_relative_to(REPO_ROOT) else calib_path_out),
                "rows": write_jsonl(calib_path_out, calib, overwrite),
            },
            "test": {
                "path": str(test_path_out.relative_to(REPO_ROOT) if test_path_out.is_relative_to(REPO_ROOT) else test_path_out),
                "rows": write_jsonl(test_path_out, test, overwrite),
            },
        }
        dataset_audit["calib_test_overlap"] = overlap_count(calib, test)
        audit["datasets"][dataset_name] = dataset_audit

    names = sorted(split_rows)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            audit["overlaps"][f"{left}.calib__{right}.calib"] = overlap_count(split_rows[left]["calib"], split_rows[right]["calib"])
            audit["overlaps"][f"{left}.test__{right}.test"] = overlap_count(split_rows[left]["test"], split_rows[right]["test"])

    audit_path = output_dir / "split_audit.json"
    if audit_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {audit_path}")
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return audit


def main() -> None:
    args = parse_args()
    config = load_config(repo_path(args.config))
    output_dir = repo_path(args.output_dir or config["output_dir"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    audit = prepare(config, output_dir, seed, args.allow_missing, args.overwrite)

    print(f"[done] wrote splits to {audit['output_dir']}")
    for name, item in audit["datasets"].items():
        calib_rows = item["outputs"]["calib"]["rows"]
        test_rows = item["outputs"]["test"]["rows"]
        overlap = item["calib_test_overlap"]
        print(f"  {name}: calib={calib_rows} test={test_rows} calib_test_overlap={overlap}")
        for source_name, source in item["sources"].items():
            if source.get("warning"):
                print(f"    warning[{source_name}]: {source['warning']}")


if __name__ == "__main__":
    main()
