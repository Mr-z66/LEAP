#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
LABEL_DIR="${LABEL_DIR:-dataset/mixed_probe_labels_fallback_second_pass}"
OUTPUT_DIR="${OUTPUT_DIR:-result/analysis_outputs/mixed_label_audits}"
DATASETS="${DATASETS:-gsm8k_calib svamp_calib math500_calib}"
MAX_EXAMPLES_PER_BUCKET="${MAX_EXAMPLES_PER_BUCKET:-20}"

cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

run_audit() {
  local key="$1"
  local label_path="${LABEL_DIR}/${key}_labels.pt"
  local output_prefix="${key}"

  if [[ ! -f "${label_path}" ]]; then
    echo "[skip] missing label file: ${label_path}"
    return 0
  fi

  echo
  echo "========== audit ${key} =========="
  echo "[input]  ${label_path}"
  echo "[output] ${OUTPUT_DIR}/${output_prefix}_*"

  python -m evaluation.audit_chunk_label_consistency \
    --label-path "${label_path}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-prefix "${output_prefix}" \
    --max-examples-per-bucket "${MAX_EXAMPLES_PER_BUCKET}"

  python - <<PY
import json
from pathlib import Path

summary_path = Path("${OUTPUT_DIR}") / "${output_prefix}_label_audit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
focus = {
    "questions": summary["questions"],
    "rows_used": summary["rows_used"],
    "error_chunk_prevalence": round(float(summary["error_chunk_prevalence"]), 4),
    "final_wrong_but_no_error_chunk": summary["final_wrong_but_no_error_chunk"],
    "final_correct_but_has_error_chunk": summary["final_correct_but_has_error_chunk"],
    "zero_to_one_label_flip_questions": summary["zero_to_one_label_flip_questions"],
    "low_judge_confidence_questions": summary["low_judge_confidence_questions"],
    "judge_parse_statuses": summary["judge_parse_statuses"],
}
print("[focus]", json.dumps(focus, ensure_ascii=True))
PY
}

for key in ${DATASETS}; do
  run_audit "${key}"
done

echo
echo "[done] audit artifacts saved under ${OUTPUT_DIR}"
