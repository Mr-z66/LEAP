#!/usr/bin/env bash
set -euo pipefail

# Mixed-probe LEAP with persistent, rollback-capable KV caches.  This is the
# fixed4 configuration used for the main comparison: every trigger gives the
# large model four chunks, with no adaptive early return.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export ROOT_DIR
export DATASETS="${DATASETS:-gsm8k_test svamp_test math500_test}"
export TRAIN_LABEL_PATH="${TRAIN_LABEL_PATH:-dataset/mixed_gsm8k_svamp_math500_calib_labels_balanced_5to1.pt}"
export ARTIFACT_PATH="${ARTIFACT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp_math500_balanced_5to1.pt}"
export SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-1.5B}"
export LARGE_MODEL_PATH_GSM8K_SVAMP="${LARGE_MODEL_PATH_GSM8K_SVAMP:-${ROOT_DIR}/models/Qwen2.5-7B}"
export LARGE_MODEL_PATH_MATH500="${LARGE_MODEL_PATH_MATH500:-${ROOT_DIR}/models/Qwen2.5-32B}"
export LARGE_MODEL_PARAMS_B_GSM8K_SVAMP="${LARGE_MODEL_PARAMS_B_GSM8K_SVAMP:-7.0}"
export LARGE_MODEL_PARAMS_B_MATH500="${LARGE_MODEL_PARAMS_B_MATH500:-32.0}"

export THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
export RUNTIME_CHUNKING="${RUNTIME_CHUNKING:-rsdmath}"
export HANDOFF_MODE="takeover"
export ADAPTIVE_LARGE_HANDOFF="0"
export LARGE_HANDOFF_CHUNKS="4"
export PERSISTENT_KV_CACHE="1"
export TRACE_TAG="${TRACE_TAG:-mixedprobe_kv_fixed4}"

echo "[config] mixed probe: ${ARTIFACT_PATH}"
echo "[config] datasets: ${DATASETS}"
echo "[config] policy: fixed4 + persistent KV cache"
echo "[config] GSM8K/SVAMP large model: ${LARGE_MODEL_PATH_GSM8K_SVAMP}"
echo "[config] MATH500 large model: ${LARGE_MODEL_PATH_MATH500}"

exec bash "${SCRIPT_DIR}/run_scheduler_eval.sh"
