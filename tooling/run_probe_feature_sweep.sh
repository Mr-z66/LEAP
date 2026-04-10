#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LABEL_PATH="${LABEL_PATH:-gsm8k_labeled_training_data_strict.pt}"
RUN_DIR="${RUN_DIR:-probe_feature_sweep_runs}"
mkdir -p "${RUN_DIR}"

declare -a FEATURE_NAMES=(
  "boundary"
  "boundary_mean"
  "boundary_mean_pos"
  "boundary_mean_uncertainty"
  "mainline_full"
  "mainline_full_boundary_drift"
)

declare -a FEATURE_SPECS=(
  "boundary"
  "boundary+mean"
  "boundary+mean+relative_position"
  "boundary+mean+final_entropy+final_margin+final_top1_prob"
  "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob"
  "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob+boundary_cosine_drift"
)

echo "Running probe feature sweep in ${REPO_ROOT}"
echo "Label path: ${LABEL_PATH}"
echo "Output dir: ${RUN_DIR}"

for i in "${!FEATURE_NAMES[@]}"; do
  name="${FEATURE_NAMES[$i]}"
  spec="${FEATURE_SPECS[$i]}"
  artifact_path="${RUN_DIR}/probe_artifact_${name}.pt"
  train_log="${RUN_DIR}/${name}_train.log"
  eval_log="${RUN_DIR}/${name}_eval.log"

  echo
  echo "=================================================="
  echo "[$((i + 1))/${#FEATURE_NAMES[@]}] Training feature set: ${name}"
  echo "Spec: ${spec}"
  echo "Artifact: ${artifact_path}"
  echo "=================================================="

  python probes/train_probe_artifact_torch.py \
    --label-path "${LABEL_PATH}" \
    --output-path "${artifact_path}" \
    --feature-key "${spec}" \
    2>&1 | tee "${train_log}"

  echo
  echo "Evaluating feature set: ${name}"
  python probes/evaluate_probe_baseline_torch.py \
    --data-path "${LABEL_PATH}" \
    --artifact-path "${artifact_path}" \
    2>&1 | tee "${eval_log}"
done

echo
echo "Feature sweep complete. Logs and artifacts are in ${RUN_DIR}"
