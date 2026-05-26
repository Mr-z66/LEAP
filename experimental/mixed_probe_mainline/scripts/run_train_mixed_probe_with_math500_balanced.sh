#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
INPUT_LABEL_PATH="${INPUT_LABEL_PATH:-dataset/mixed_gsm8k_svamp_math500_calib_labels.pt}"
BALANCED_LABEL_PATH="${BALANCED_LABEL_PATH:-dataset/mixed_gsm8k_svamp_math500_calib_labels_balanced_3to1.pt}"
BALANCED_SUMMARY_PATH="${BALANCED_SUMMARY_PATH:-result/analysis_outputs/mixed_gsm8k_svamp_math500_balanced_3to1_summary.json}"
OUTPUT_PATH="${OUTPUT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp_math500_balanced_3to1.pt}"
FEATURE_KEY="${FEATURE_KEY:-boundary+mean}"
NEG_TO_POS_RATIO="${NEG_TO_POS_RATIO:-3.0}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p result/artifacts result/logs/mixed_probe_mainline result/analysis_outputs
LOG_PATH="result/logs/mixed_probe_mainline/train_mixed_probe_with_math500_balanced_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${INPUT_LABEL_PATH}" ]]; then
  echo "[error] missing input labels: ${INPUT_LABEL_PATH}" >&2
  echo "[hint] run first:" >&2
  echo "  bash experimental/mixed_probe_mainline/scripts/run_merge_mixed_with_math500.sh" >&2
  exit 2
fi

echo "[balance] input=${INPUT_LABEL_PATH}"
echo "[balance] output=${BALANCED_LABEL_PATH}"
echo "[balance] neg_to_pos_ratio=${NEG_TO_POS_RATIO}"
echo "[train] output=${OUTPUT_PATH}"
echo "[train] feature=${FEATURE_KEY}"
echo "[log]   ${LOG_PATH}"

python experimental/mixed_probe_mainline/scripts/make_balanced_labeled_dataset.py \
  --input-path "${INPUT_LABEL_PATH}" \
  --output-path "${BALANCED_LABEL_PATH}" \
  --summary-path "${BALANCED_SUMMARY_PATH}" \
  --neg-to-pos-ratio "${NEG_TO_POS_RATIO}" \
  --by-domain \
  --drop-unlabeled \
  2>&1 | tee "${LOG_PATH}"

echo
python -m core_package.probes.train_probe_artifact_torch \
  --label-path "${BALANCED_LABEL_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --feature-key "${FEATURE_KEY}" \
  --hidden-layers 128,32 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3 \
  2>&1 | tee -a "${LOG_PATH}"

echo
python -m core_package.probes.evaluate_probe_baseline_torch \
  --data-path "${BALANCED_LABEL_PATH}" \
  --artifact-path "${OUTPUT_PATH}" \
  --threshold-grid "${THRESHOLD_GRID}" \
  2>&1 | tee -a "${LOG_PATH}"

echo
echo "[done] balanced labels saved to ${BALANCED_LABEL_PATH}"
echo "[done] balanced probe artifact saved to ${OUTPUT_PATH}"
