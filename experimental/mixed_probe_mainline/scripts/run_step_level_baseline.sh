#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/LEAP_data}"
LABEL_DIR="${LABEL_DIR:-${DATA_ROOT}/dataset/mixed_probe_labels_rsdstep_second_pass}"
TRAJ_DIR="${TRAJ_DIR:-${DATA_ROOT}/dataset/mixed_probe_trajectories-old}"
DATASETS_CALIB="${DATASETS_CALIB:-gsm8k_calib,svamp_calib,math500_calib}"
DATASETS_TEST="${DATASETS_TEST:-gsm8k_test svamp_test math500_test}"

MERGED_LABEL_PATH="${MERGED_LABEL_PATH:-${DATA_ROOT}/dataset/mixed_gsm8k_svamp_math500_rsdstep_calib_labels.pt}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-${DATA_ROOT}/result/analysis_outputs/rsdstep_mixed_merge_summary.json}"
BALANCED_LABEL_PATH="${BALANCED_LABEL_PATH:-${DATA_ROOT}/dataset/mixed_gsm8k_svamp_math500_rsdstep_calib_labels_balanced_5to1.pt}"
BALANCED_SUMMARY_PATH="${BALANCED_SUMMARY_PATH:-${DATA_ROOT}/result/analysis_outputs/rsdstep_mixed_balanced_5to1_summary.json}"
ARTIFACT_PATH="${ARTIFACT_PATH:-${ROOT_DIR}/result/artifacts/probe_artifact_mixed_gsm8k_svamp_math500_rsdstep_balanced_5to1.pt}"

NEG_TO_POS_RATIO="${NEG_TO_POS_RATIO:-5.0}"
FEATURE_KEY="${FEATURE_KEY:-boundary+mean}"
THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
TRACE_TAG="${TRACE_TAG:-mixed_rsdstep_5to1}"
STEP_FORCE_TOKENS="${STEP_FORCE_TOKENS:-2048}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p \
  "${DATA_ROOT}/result/analysis_outputs" \
  "${DATA_ROOT}/result/traces" \
  result/artifacts \
  result/logs/mixed_probe_mainline

echo "[step-baseline] label_dir=${LABEL_DIR}"
echo "[step-baseline] traj_dir=${TRAJ_DIR}"
echo "[step-baseline] merged=${MERGED_LABEL_PATH}"
echo "[step-baseline] balanced=${BALANCED_LABEL_PATH}"
echo "[step-baseline] artifact=${ARTIFACT_PATH}"
echo "[step-baseline] thresholds=${THRESHOLDS}"
echo "[step-baseline] force_tokens=${STEP_FORCE_TOKENS}"

python experimental/mixed_probe_mainline/scripts/merge_labeled_datasets.py \
  --input-dir "${LABEL_DIR}" \
  --datasets "${DATASETS_CALIB}" \
  --output-path "${MERGED_LABEL_PATH}" \
  --summary-path "${MERGE_SUMMARY_PATH}"

python experimental/mixed_probe_mainline/scripts/make_balanced_labeled_dataset.py \
  --input-path "${MERGED_LABEL_PATH}" \
  --output-path "${BALANCED_LABEL_PATH}" \
  --summary-path "${BALANCED_SUMMARY_PATH}" \
  --neg-to-pos-ratio "${NEG_TO_POS_RATIO}" \
  --by-domain \
  --drop-unlabeled

python -m core_package.probes.train_probe_artifact_torch \
  --label-path "${BALANCED_LABEL_PATH}" \
  --output-path "${ARTIFACT_PATH}" \
  --feature-key "${FEATURE_KEY}" \
  --hidden-layers 128,32 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3

python -m core_package.probes.evaluate_probe_baseline_torch \
  --data-path "${BALANCED_LABEL_PATH}" \
  --artifact-path "${ARTIFACT_PATH}" \
  --threshold-grid "${THRESHOLDS}"

TRAIN_LABEL_PATH="${BALANCED_LABEL_PATH}" \
TRAJ_DIR="${TRAJ_DIR}" \
ARTIFACT_PATH="${ARTIFACT_PATH}" \
DATASETS="${DATASETS_TEST}" \
THRESHOLDS="${THRESHOLDS}" \
RUNTIME_CHUNKING="rsd_step" \
REWRITE_STEP_FORCE_TOKENS="${STEP_FORCE_TOKENS}" \
TRACE_TAG="${TRACE_TAG}" \
bash experimental/mixed_probe_mainline/scripts/run_scheduler_eval.sh

echo
echo "[done] step-level baseline finished"
echo "[done] traces: result/traces/observe_rollback_traces_${TRACE_TAG}_*.json"
