#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
INPUT_DIR="${INPUT_DIR:-dataset/mixed_probe_labels_fallback_second_pass}"
OUTPUT_PATH="${OUTPUT_PATH:-dataset/mixed_gsm8k_svamp_math500_calib_labels.pt}"
SUMMARY_PATH="${SUMMARY_PATH:-result/analysis_outputs/mixed_gsm8k_svamp_math500_merge_summary.json}"
DATASETS="${DATASETS:-gsm8k_calib,svamp_calib,math500_calib}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python experimental/mixed_probe_mainline/scripts/merge_labeled_datasets.py \
  --input-dir "${INPUT_DIR}" \
  --datasets "${DATASETS}" \
  --output-path "${OUTPUT_PATH}" \
  --summary-path "${SUMMARY_PATH}"

echo
echo "[done] mixed+math500 labels saved to ${OUTPUT_PATH}"
echo "[done] summary saved to ${SUMMARY_PATH}"
