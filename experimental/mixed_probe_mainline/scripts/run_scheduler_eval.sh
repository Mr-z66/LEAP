#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
TRAIN_LABEL_PATH="${TRAIN_LABEL_PATH:-dataset/mixed_gsm8k_svamp_calib_labels.pt}"
TRAJ_DIR="${TRAJ_DIR:-dataset/mixed_probe_trajectories_fallback}"
ARTIFACT_PATH="${ARTIFACT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp.pt}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH_GSM8K_SVAMP="${LARGE_MODEL_PATH_GSM8K_SVAMP:-/root/autodl-tmp/models/Qwen2.5-7B}"
LARGE_MODEL_PATH_MATH500="${LARGE_MODEL_PATH_MATH500:-/root/autodl-tmp/models/Qwen2.5-32B}"
THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45}"
DATASETS="${DATASETS:-gsm8k_test svamp_test}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p result/traces result/logs/mixed_probe_mainline

run_gsm8k() {
  local eval_path="${TRAJ_DIR}/gsm8k_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_mixed_probe_gsm8k_test.json"
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
    --max-handoffs 2 \
    --large-handoff-chunks 2 \
    --adaptive-large-handoff \
    --min-large-handoff-chunks 4 \
    --max-adaptive-large-handoff-chunks 4 \
    --handoff-recovery-threshold 0.25 \
    --cooldown-chunks 2 \
    --answer-type gsm8k_boxed_numeric \
    --small-model-params-b 1.5 \
    --large-model-params-b 7.0 \
    --trace-export-path "${trace_path}" \
    2>&1 | tee "${log_path}"
}

run_svamp() {
  local eval_path="${TRAJ_DIR}/svamp_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_mixed_probe_svamp_test.json"
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
    --max-handoffs 2 \
    --large-handoff-chunks 2 \
    --adaptive-large-handoff \
    --min-large-handoff-chunks 4 \
    --max-adaptive-large-handoff-chunks 4 \
    --handoff-recovery-threshold 0.25 \
    --cooldown-chunks 2 \
    --answer-type svamp_boxed_numeric \
    --small-model-params-b 1.5 \
    --large-model-params-b 7.0 \
    --trace-export-path "${trace_path}" \
    2>&1 | tee "${log_path}"
}

run_math500() {
  local eval_path="${TRAJ_DIR}/math500_test_300_15b.pt"
  local trace_path="result/traces/observe_rollback_traces_mixed_probe_math500_test.json"
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
    --max-handoffs 2 \
    --large-handoff-chunks 2 \
    --adaptive-large-handoff \
    --min-large-handoff-chunks 4 \
    --max-adaptive-large-handoff-chunks 4 \
    --handoff-recovery-threshold 0.25 \
    --cooldown-chunks 2 \
    --answer-type math500_qwen_boxed \
    --small-model-params-b 1.5 \
    --large-model-params-b 32.0 \
    --trace-export-path "${trace_path}" \
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
