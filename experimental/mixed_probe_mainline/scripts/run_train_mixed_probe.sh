#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
MERGED_LABEL_PATH="${MERGED_LABEL_PATH:-dataset/mixed_gsm8k_svamp_calib_labels.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp.pt}"
FEATURE_KEY="${FEATURE_KEY:-boundary+mean}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p result/artifacts result/logs/mixed_probe_mainline
LOG_PATH="result/logs/mixed_probe_mainline/train_mixed_probe_$(date +%Y%m%d_%H%M%S).log"

echo "[train] labels=${MERGED_LABEL_PATH}"
echo "[train] output=${OUTPUT_PATH}"
echo "[train] feature=${FEATURE_KEY}"
echo "[log]   ${LOG_PATH}"

if [[ ! -f "${MERGED_LABEL_PATH}" ]]; then
  echo "[error] missing merged labels: ${MERGED_LABEL_PATH}" >&2
  echo "[hint] run first:" >&2
  echo "  python experimental/mixed_probe_mainline/scripts/merge_labeled_datasets.py" >&2
  exit 2
fi

python -m core_package.probes.train_probe_artifact_torch \
  --label-path "${MERGED_LABEL_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --feature-key "${FEATURE_KEY}" \
  --hidden-layers 128,32 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3 \
  2>&1 | tee "${LOG_PATH}"

echo
python -m core_package.probes.evaluate_probe_baseline_torch \
  --data-path "${MERGED_LABEL_PATH}" \
  --artifact-path "${OUTPUT_PATH}" \
  --threshold-grid "${THRESHOLD_GRID}" \
  2>&1 | tee -a "${LOG_PATH}"

echo
echo "[done] mixed probe artifact saved to ${OUTPUT_PATH}"
