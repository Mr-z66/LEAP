#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
TRAIN_LABEL_PATH="${TRAIN_LABEL_PATH:-dataset/mixed_gsm8k_svamp_calib_labels.pt}"
TRAJ_DIR="${TRAJ_DIR:-dataset/mixed_probe_trajectories_fallback}"
ARTIFACT_PATH="${ARTIFACT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp.pt}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH_GSM8K_SVAMP="${LARGE_MODEL_PATH_GSM8K_SVAMP:-/root/autodl-tmp/models/Qwen2.5-7B}"
LARGE_MODEL_PATH_MATH500="${LARGE_MODEL_PATH_MATH500:-/root/autodl-tmp/models/Qwen2.5-7B}"
LARGE_MODEL_PARAMS_B_MATH500="${LARGE_MODEL_PARAMS_B_MATH500:-7.0}"
THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45}"
DATASETS="${DATASETS:-gsm8k_test svamp_test}"
RUNTIME_CHUNKING="${RUNTIME_CHUNKING:-rsdmath}"
HANDOFF_MODE="${HANDOFF_MODE:-takeover}"
ADAPTIVE_LARGE_HANDOFF="${ADAPTIVE_LARGE_HANDOFF:-1}"
MAX_HANDOFFS="${MAX_HANDOFFS:-2}"
LARGE_HANDOFF_CHUNKS="${LARGE_HANDOFF_CHUNKS:-2}"
MIN_LARGE_HANDOFF_CHUNKS="${MIN_LARGE_HANDOFF_CHUNKS:-4}"
MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS="${MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS:-4}"
HANDOFF_RECOVERY_THRESHOLD="${HANDOFF_RECOVERY_THRESHOLD:-0.25}"
COOLDOWN_CHUNKS="${COOLDOWN_CHUNKS:-2}"
TRACE_TAG="${TRACE_TAG:-mixed_probe}"
EXTRA_SCHEDULER_ARGS="${EXTRA_SCHEDULER_ARGS:-}"
REWRITE_STEP_MIN_TOKENS="${REWRITE_STEP_MIN_TOKENS:-12}"
REWRITE_STEP_TARGET_TOKENS="${REWRITE_STEP_TARGET_TOKENS:-64}"
REWRITE_STEP_FORCE_TOKENS="${REWRITE_STEP_FORCE_TOKENS:-160}"
REWRITE_STEP_BOUNDARY_MODE="${REWRITE_STEP_BOUNDARY_MODE:-auto}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p result/traces result/logs/mixed_probe_mainline

ADAPTIVE_LARGE_HANDOFF_ARGS=""
if [[ "${ADAPTIVE_LARGE_HANDOFF}" == "1" ]]; then
  ADAPTIVE_LARGE_HANDOFF_ARGS="--adaptive-large-handoff"
fi

if [[ ! -f "${TRAIN_LABEL_PATH}" ]]; then
  echo "[error] missing merged training labels: ${TRAIN_LABEL_PATH}" >&2
  echo "[hint] run first:" >&2
  echo "  python experimental/mixed_probe_mainline/scripts/merge_labeled_datasets.py" >&2
  exit 2
fi

if [[ ! -f "${ARTIFACT_PATH}" ]]; then
  echo "[error] missing probe artifact: ${ARTIFACT_PATH}" >&2
  echo "[hint] run first:" >&2
  echo "  bash experimental/mixed_probe_mainline/scripts/run_train_mixed_probe.sh" >&2
  exit 2
fi

run_gsm8k() {
  local eval_path="${TRAJ_DIR}/gsm8k_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_${TRACE_TAG}_gsm8k_test.json"
  local log_path="result/logs/mixed_probe_mainline/scheduler_gsm8k_test_$(date +%Y%m%d_%H%M%S).log"
  [[ -f "${eval_path}" ]] || { echo "[skip] missing trajectory: ${eval_path}"; return 0; }
  python -m core_package.schedulers.simulate_observe_rollback_scheduler \
    --label-path "${TRAIN_LABEL_PATH}" \
    --eval-data-path "${eval_path}" \
    --probe-artifact-path "${ARTIFACT_PATH}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH_GSM8K_SVAMP}" \
    --large-backend hf \
    --thresholds "${THRESHOLDS}" \
    --num-test-questions 300 \
    --max-new-tokens 768 \
    --runtime-chunking "${RUNTIME_CHUNKING}" \
    --max-handoffs "${MAX_HANDOFFS}" \
    --handoff-mode "${HANDOFF_MODE}" \
    --rewrite-step-min-tokens "${REWRITE_STEP_MIN_TOKENS}" \
    --rewrite-step-target-tokens "${REWRITE_STEP_TARGET_TOKENS}" \
    --rewrite-step-force-tokens "${REWRITE_STEP_FORCE_TOKENS}" \
    --rewrite-step-boundary-mode "${REWRITE_STEP_BOUNDARY_MODE}" \
    --large-handoff-chunks "${LARGE_HANDOFF_CHUNKS}" \
    ${ADAPTIVE_LARGE_HANDOFF_ARGS} \
    --min-large-handoff-chunks "${MIN_LARGE_HANDOFF_CHUNKS}" \
    --max-adaptive-large-handoff-chunks "${MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS}" \
    --handoff-recovery-threshold "${HANDOFF_RECOVERY_THRESHOLD}" \
    --cooldown-chunks "${COOLDOWN_CHUNKS}" \
    --answer-type gsm8k_boxed_numeric \
    --small-model-params-b 1.5 \
    --large-model-params-b 7.0 \
    --trace-export-path "${trace_path}" \
    ${EXTRA_SCHEDULER_ARGS} \
    2>&1 | tee "${log_path}"
}

run_svamp() {
  local eval_path="${TRAJ_DIR}/svamp_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_${TRACE_TAG}_svamp_test.json"
  local log_path="result/logs/mixed_probe_mainline/scheduler_svamp_test_$(date +%Y%m%d_%H%M%S).log"
  [[ -f "${eval_path}" ]] || { echo "[skip] missing trajectory: ${eval_path}"; return 0; }
  python -m core_package.schedulers.simulate_observe_rollback_scheduler_svamp \
    --label-path "${TRAIN_LABEL_PATH}" \
    --eval-data-path "${eval_path}" \
    --probe-artifact-path "${ARTIFACT_PATH}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH_GSM8K_SVAMP}" \
    --large-backend hf \
    --thresholds "${THRESHOLDS}" \
    --num-test-questions 300 \
    --max-new-tokens 512 \
    --runtime-chunking "${RUNTIME_CHUNKING}" \
    --max-handoffs "${MAX_HANDOFFS}" \
    --handoff-mode "${HANDOFF_MODE}" \
    --rewrite-step-min-tokens "${REWRITE_STEP_MIN_TOKENS}" \
    --rewrite-step-target-tokens "${REWRITE_STEP_TARGET_TOKENS}" \
    --rewrite-step-force-tokens "${REWRITE_STEP_FORCE_TOKENS}" \
    --rewrite-step-boundary-mode "${REWRITE_STEP_BOUNDARY_MODE}" \
    --large-handoff-chunks "${LARGE_HANDOFF_CHUNKS}" \
    ${ADAPTIVE_LARGE_HANDOFF_ARGS} \
    --min-large-handoff-chunks "${MIN_LARGE_HANDOFF_CHUNKS}" \
    --max-adaptive-large-handoff-chunks "${MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS}" \
    --handoff-recovery-threshold "${HANDOFF_RECOVERY_THRESHOLD}" \
    --cooldown-chunks "${COOLDOWN_CHUNKS}" \
    --answer-type svamp_boxed_numeric \
    --small-model-params-b 1.5 \
    --large-model-params-b 7.0 \
    --trace-export-path "${trace_path}" \
    ${EXTRA_SCHEDULER_ARGS} \
    2>&1 | tee "${log_path}"
}

run_math500() {
  local eval_path="${TRAJ_DIR}/math500_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_${TRACE_TAG}_math500_test.json"
  local log_path="result/logs/mixed_probe_mainline/scheduler_math500_test_$(date +%Y%m%d_%H%M%S).log"
  [[ -f "${eval_path}" ]] || { echo "[skip] missing trajectory: ${eval_path}"; return 0; }
  python -m core_package.schedulers.simulate_observe_rollback_scheduler \
    --label-path "${TRAIN_LABEL_PATH}" \
    --eval-data-path "${eval_path}" \
    --probe-artifact-path "${ARTIFACT_PATH}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH_MATH500}" \
    --large-backend hf \
    --thresholds "${THRESHOLDS}" \
    --num-test-questions 300 \
    --max-new-tokens 1024 \
    --runtime-chunking "${RUNTIME_CHUNKING}" \
    --max-handoffs "${MAX_HANDOFFS}" \
    --handoff-mode "${HANDOFF_MODE}" \
    --rewrite-step-min-tokens "${REWRITE_STEP_MIN_TOKENS}" \
    --rewrite-step-target-tokens "${REWRITE_STEP_TARGET_TOKENS}" \
    --rewrite-step-force-tokens "${REWRITE_STEP_FORCE_TOKENS}" \
    --rewrite-step-boundary-mode "${REWRITE_STEP_BOUNDARY_MODE}" \
    --large-handoff-chunks "${LARGE_HANDOFF_CHUNKS}" \
    ${ADAPTIVE_LARGE_HANDOFF_ARGS} \
    --min-large-handoff-chunks "${MIN_LARGE_HANDOFF_CHUNKS}" \
    --max-adaptive-large-handoff-chunks "${MAX_ADAPTIVE_LARGE_HANDOFF_CHUNKS}" \
    --handoff-recovery-threshold "${HANDOFF_RECOVERY_THRESHOLD}" \
    --cooldown-chunks "${COOLDOWN_CHUNKS}" \
    --answer-type math500_qwen_boxed \
    --small-model-params-b 1.5 \
    --large-model-params-b "${LARGE_MODEL_PARAMS_B_MATH500}" \
    --trace-export-path "${trace_path}" \
    ${EXTRA_SCHEDULER_ARGS} \
    2>&1 | tee "${log_path}"
}

for key in ${DATASETS}; do
  case "${key}" in
    gsm8k_test) run_gsm8k ;;
    svamp_test) run_svamp ;;
    math500_test) run_math500 ;;
    *) echo "[skip] unsupported dataset key: ${key}" ;;
  esac
done

echo
echo "[done] mixed scheduler eval finished"
