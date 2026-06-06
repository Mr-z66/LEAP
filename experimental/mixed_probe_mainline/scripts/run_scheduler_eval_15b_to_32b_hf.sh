#!/usr/bin/env bash
set -euo pipefail

# Re-run the mixed-probe fixed4 scheduler with a 32B large model.
# This deliberately reuses the same probe artifact, trajectories, thresholds,
# and routing policy as the existing 1.5B-to-7B mixed mainline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH_32B="${LARGE_MODEL_PATH_32B:-${ROOT_DIR}/models/Qwen2.5-32B}"

export ROOT_DIR
export SMALL_MODEL_PATH
export DATASETS="${DATASETS:-gsm8k_test svamp_test math500_test}"
export TRAIN_LABEL_PATH="${TRAIN_LABEL_PATH:-dataset/mixed_gsm8k_svamp_math500_calib_labels_balanced_5to1.pt}"
export ARTIFACT_PATH="${ARTIFACT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp_math500_balanced_5to1.pt}"
export LARGE_MODEL_PATH_GSM8K_SVAMP="${LARGE_MODEL_PATH_GSM8K_SVAMP:-${LARGE_MODEL_PATH_32B}}"
export LARGE_MODEL_PATH_MATH500="${LARGE_MODEL_PATH_MATH500:-${LARGE_MODEL_PATH_32B}}"
export LARGE_MODEL_PARAMS_B_GSM8K_SVAMP="${LARGE_MODEL_PARAMS_B_GSM8K_SVAMP:-32.0}"
export LARGE_MODEL_PARAMS_B_MATH500="${LARGE_MODEL_PARAMS_B_MATH500:-32.0}"

# Match the current mixed fixed4 sweep.
export THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
export RUNTIME_CHUNKING="${RUNTIME_CHUNKING:-rsdmath}"
export HANDOFF_MODE="${HANDOFF_MODE:-takeover}"
export ADAPTIVE_LARGE_HANDOFF="${ADAPTIVE_LARGE_HANDOFF:-1}"
export MAX_HANDOFFS="${MAX_HANDOFFS:-2}"
export LARGE_HANDOFF_CHUNKS="${LARGE_HANDOFF_CHUNKS:-2}"
export MIN_LARGE_HANDOFF_CHUNKS="${MIN_LARGE_HANDOFF_CHUNKS:-4}"
export MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS="${MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS:-4}"
export HANDOFF_RECOVERY_THRESHOLD="${HANDOFF_RECOVERY_THRESHOLD:-0.25}"
export COOLDOWN_CHUNKS="${COOLDOWN_CHUNKS:-2}"
export TRACE_TAG="${TRACE_TAG:-mixed_fixed4_sweep015_050_15b_to_32b_hf}"

echo "[config] probe=${ARTIFACT_PATH}"
echo "[config] train_labels=${TRAIN_LABEL_PATH}"
echo "[config] small_model=${SMALL_MODEL_PATH}"
echo "[config] large_model=${LARGE_MODEL_PATH_32B}"
echo "[config] datasets=${DATASETS}"
echo "[config] thresholds=${THRESHOLDS}"
echo "[config] trace_tag=${TRACE_TAG}"

exec bash "${ROOT_DIR}/experimental/mixed_probe_mainline/scripts/run_scheduler_eval.sh"
