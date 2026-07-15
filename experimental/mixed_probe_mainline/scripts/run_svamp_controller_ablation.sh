#!/usr/bin/env bash
set -euo pipefail

# SVAMP controller ablation for the paper-facing LEAP configuration.
# All variants share the same data, probe, threshold, models, decoding budget,
# and decode-cost accounting. Persistent KV is intentionally disabled for all
# variants because the current adaptive controller does not support it and KV
# does not change the decoded trajectory or the primary decode-cost metric.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-care_env}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/LEAP_data}"

TRAIN_LABEL_PATH="${TRAIN_LABEL_PATH:-${DATA_ROOT}/dataset/mixed_gsm8k_svamp_math500_calib_labels_balanced_5to1.pt}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_ROOT}/dataset/mixed_probe_trajectories-old/svamp_test_300_15b.pt}"
ARTIFACT_PATH="${ARTIFACT_PATH:-${DATA_ROOT}/result/artifacts/probe_artifact_mixed_gsm8k_svamp_math500_balanced_5to1.pt}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B}"

THRESHOLD="${THRESHOLD:-0.20}"
NUM_TEST_QUESTIONS="${NUM_TEST_QUESTIONS:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_HANDOFFS="${MAX_HANDOFFS:-2}"
COOLDOWN_CHUNKS="${COOLDOWN_CHUNKS:-2}"
RECOVERY_THRESHOLD="${RECOVERY_THRESHOLD:-0.25}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-55}"
LARGE_ONLY_COST="${LARGE_ONLY_COST:-1445.4}"
INCLUDE_EXTENDED="${INCLUDE_EXTENDED:-0}"

TRACE_DIR="${TRACE_DIR:-${ROOT_DIR}/result/traces/controller_ablation_svamp}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/result/logs/controller_ablation_svamp}"
SUMMARY_DIR="${SUMMARY_DIR:-${ROOT_DIR}/result/analysis_outputs/controller_ablation_svamp}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "${TRACE_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}"

for required in "${TRAIN_LABEL_PATH}" "${EVAL_DATA_PATH}" "${ARTIFACT_PATH}"; do
  [[ -f "${required}" ]] || { echo "[error] missing required file: ${required}" >&2; exit 2; }
done
for required in "${SMALL_MODEL_PATH}" "${LARGE_MODEL_PATH}"; do
  [[ -d "${required}" ]] || { echo "[error] missing model directory: ${required}" >&2; exit 2; }
done

DEFAULT_CASES="no_intervention no_rollback slm_repair no_return fixed1 fixed2 fixed4 adaptive2to4"
if [[ "${INCLUDE_EXTENDED}" == "1" ]]; then
  DEFAULT_CASES+=" fixed4_h1 fixed4_h4 fixed4_no_cooldown"
fi
read -r -a CASE_LIST <<< "${CASES:-${DEFAULT_CASES}}"

run_case() {
  local name="$1"
  local mode="standard"
  local repair_chunks=4
  local max_handoffs="${MAX_HANDOFFS}"
  local cooldown="${COOLDOWN_CHUNKS}"
  local adaptive=0

  case "${name}" in
    no_intervention) mode="no_intervention"; max_handoffs=0 ;;
    no_rollback) mode="no_rollback" ;;
    slm_repair) mode="slm_repair" ;;
    no_return) mode="no_return" ;;
    fixed1) repair_chunks=1 ;;
    fixed2) repair_chunks=2 ;;
    fixed4) repair_chunks=4 ;;
    adaptive2to4) adaptive=1; repair_chunks=4 ;;
    fixed4_h1) repair_chunks=4; max_handoffs=1 ;;
    fixed4_h4) repair_chunks=4; max_handoffs=4 ;;
    fixed4_no_cooldown) repair_chunks=4; cooldown=0 ;;
    *) echo "[error] unknown ablation case: ${name}" >&2; return 2 ;;
  esac

  local trace_path="${TRACE_DIR}/${name}.json"
  local log_path="${LOG_DIR}/${name}.log"
  local adaptive_args=()
  if [[ "${adaptive}" == "1" ]]; then
    adaptive_args=(
      --adaptive-large-handoff
      --min-large-handoff-chunks 2
      --max-adaptive-large-handoff-chunks 4
      --handoff-recovery-threshold "${RECOVERY_THRESHOLD}"
    )
  fi

  echo "[run] ${name} | mode=${mode} fixed=${repair_chunks} handoffs=${max_handoffs} cooldown=${cooldown}"
  python -m core_package.schedulers.simulate_observe_rollback_scheduler \
    --label-path "${TRAIN_LABEL_PATH}" \
    --eval-data-path "${EVAL_DATA_PATH}" \
    --probe-artifact-path "${ARTIFACT_PATH}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH}" \
    --large-backend hf \
    --thresholds "${THRESHOLD}" \
    --num-test-questions "${NUM_TEST_QUESTIONS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --runtime-chunking rsdmath \
    --max-handoffs "${max_handoffs}" \
    --handoff-mode takeover \
    --controller-ablation-mode "${mode}" \
    --rewrite-step-min-tokens 12 \
    --rewrite-step-target-tokens 64 \
    --rewrite-step-force-tokens 160 \
    --rewrite-step-boundary-mode auto \
    --large-handoff-chunks "${repair_chunks}" \
    --cooldown-chunks "${cooldown}" \
    --answer-type svamp_boxed_numeric \
    --small-model-params-b 1.5 \
    --large-model-params-b 7.0 \
    --trace-export-path "${trace_path}" \
    "${adaptive_args[@]}" \
    2>&1 | tee "${log_path}"
}

echo "[config] threshold=${THRESHOLD} questions=${NUM_TEST_QUESTIONS} max_tokens=${MAX_NEW_TOKENS}"
echo "[config] probe=${ARTIFACT_PATH}"
echo "[config] eval=${EVAL_DATA_PATH}"
echo "[config] cases=${CASE_LIST[*]}"

for name in "${CASE_LIST[@]}"; do
  run_case "${name}"
done

python experimental/mixed_probe_mainline/scripts/summarize_svamp_controller_ablation.py \
  --trace-dir "${TRACE_DIR}" \
  --output-csv "${SUMMARY_DIR}/controller_ablation_svamp.csv" \
  --output-md "${SUMMARY_DIR}/controller_ablation_svamp.md" \
  --large-only-cost "${LARGE_ONLY_COST}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${BOOTSTRAP_SEED}"

echo "[done] traces: ${TRACE_DIR}"
echo "[done] summary: ${SUMMARY_DIR}/controller_ablation_svamp.csv"
