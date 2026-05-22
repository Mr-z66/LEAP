#!/usr/bin/env bash
set -euo pipefail

# Clean LEAP rerun for GSM8K and SVAMP.
# This script keeps probe-training labels and scheduler evaluation sets separate:
#   label/train: dataset/*_labeled_training_data_strict_boxed.pt
#   eval/test:   hidden states rebuilt from dataset/gsm8k_test_300_from_pt.jsonl
#                and dataset/svamp/test.jsonl

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-care_env}"

SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-models/Qwen2.5-7B}"
if [[ ! -d "${SMALL_MODEL_PATH}" && -d /root/autodl-tmp/models/Qwen2.5-1.5B ]]; then
  SMALL_MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-1.5B"
fi
if [[ ! -d "${LARGE_MODEL_PATH}" && -d /root/autodl-tmp/models/Qwen2.5-7B ]]; then
  LARGE_MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-7B"
fi

RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_SVAMP="${RUN_SVAMP:-1}"
SKIP_BUILD_EVAL="${SKIP_BUILD_EVAL:-0}"
SKIP_TRAIN_PROBE="${SKIP_TRAIN_PROBE:-0}"

THRESHOLDS="${THRESHOLDS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45}"
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-768}"
SVAMP_MAX_NEW_TOKENS="${SVAMP_MAX_NEW_TOKENS:-512}"

GSM8K_LABEL="${GSM8K_LABEL:-dataset/gsm8k_labeled_training_data_strict_boxed.pt}"
GSM8K_TEST_JSONL="${GSM8K_TEST_JSONL:-dataset/gsm8k_test_300_from_pt.jsonl}"
GSM8K_EVAL_PT="${GSM8K_EVAL_PT:-dataset/gsm8k_test_300_15b_hidden_states_boxed_clean.pt}"
GSM8K_PROBE="${GSM8K_PROBE:-result/artifacts/probe_artifact_torch_gsm8k_boxed_clean.pt}"
GSM8K_SMALL_BASELINE="${GSM8K_SMALL_BASELINE:-result/analysis_outputs/qwen25_15b_only_gsm8k_test_boxed.json}"
GSM8K_TRACE="${GSM8K_TRACE:-result/traces/observe_rollback_traces_gsm8k_test300_15b_to_7b_boxed_clean_sweep.json}"

SVAMP_LABEL="${SVAMP_LABEL:-dataset/svamp_labeled_training_data_strict_boxed.pt}"
SVAMP_TEST_JSONL="${SVAMP_TEST_JSONL:-dataset/svamp/test.jsonl}"
SVAMP_EVAL_PT="${SVAMP_EVAL_PT:-dataset/svamp_test_300_15b_hidden_states_boxed_clean.pt}"
SVAMP_PROBE="${SVAMP_PROBE:-result/artifacts/probe_artifact_torch_svamp_boxed_clean.pt}"
SVAMP_SMALL_BASELINE="${SVAMP_SMALL_BASELINE:-result/analysis_outputs/qwen25_15b_only_svamp_test_boxed_hf_rerun.json}"
SVAMP_TRACE="${SVAMP_TRACE:-result/traces/observe_rollback_traces_svamp_test300_15b_to_7b_boxed_clean_sweep.json}"

mkdir -p result/artifacts result/traces result/logs dataset
LOG_DIR="${LOG_DIR:-result/logs/clean_rerun_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[config] ROOT_DIR=${ROOT_DIR}"
echo "[config] SMALL_MODEL_PATH=${SMALL_MODEL_PATH}"
echo "[config] LARGE_MODEL_PATH=${LARGE_MODEL_PATH}"
echo "[config] THRESHOLDS=${THRESHOLDS}"
echo "[config] LOG_DIR=${LOG_DIR}"

check_no_overlap() {
  local label_path="$1"
  local eval_path="$2"
  local name="$3"
  python - "$label_path" "$eval_path" "$name" <<'PY'
import sys
import torch

label_path, eval_path, name = sys.argv[1:4]

def norm(text):
    return " ".join(str(text).strip().split())

def get_question(row):
    if not isinstance(row, dict):
        return None
    for key in ("question", "problem", "prompt", "input", "question_concat"):
        if key in row:
            return norm(row[key])
    if "Body" in row and "Question" in row:
        return norm(str(row["Body"]) + " " + str(row["Question"]))
    return None

label_rows = torch.load(label_path, map_location="cpu", weights_only=False)
eval_rows = torch.load(eval_path, map_location="cpu", weights_only=False)
label_qs = {get_question(row) for row in label_rows if get_question(row)}
eval_qs = {get_question(row) for row in eval_rows if get_question(row)}
overlap = label_qs & eval_qs
print(f"[overlap:{name}] label_questions={len(label_qs)} eval_questions={len(eval_qs)} overlap={len(overlap)}")
if overlap:
    print(f"[overlap:{name}] ERROR: train/eval overlap detected. First example:")
    print(next(iter(overlap))[:500])
    raise SystemExit(2)
PY
}

run_gsm8k() {
  echo "========== GSM8K clean rerun =========="
  test -f "${GSM8K_LABEL}"
  test -f "${GSM8K_TEST_JSONL}"

  if [[ "${SKIP_BUILD_EVAL}" != "1" ]]; then
    echo "[gsm8k] build test-only hidden states -> ${GSM8K_EVAL_PT}"
    python -m core_package.pipelines.build_dataset \
      --dataset-name jsonl \
      --input-path "${GSM8K_TEST_JSONL}" \
      --question-field question \
      --answer-field answer \
      --answer-type gsm8k_boxed_numeric \
      --num-samples 300 \
      --model-path "${SMALL_MODEL_PATH}" \
      --save-path "${GSM8K_EVAL_PT}" \
      --max-new-tokens "${GSM8K_MAX_NEW_TOKENS}" \
      2>&1 | tee "${LOG_DIR}/01_build_gsm8k_test_hidden.log"
  fi

  check_no_overlap "${GSM8K_LABEL}" "${GSM8K_EVAL_PT}" "gsm8k"

  if [[ "${SKIP_TRAIN_PROBE}" != "1" ]]; then
    echo "[gsm8k] train probe artifact -> ${GSM8K_PROBE}"
    python -m core_package.probes.train_probe_artifact_torch \
      --label-path "${GSM8K_LABEL}" \
      --output-path "${GSM8K_PROBE}" \
      2>&1 | tee "${LOG_DIR}/02_train_gsm8k_probe.log"
  fi

  local baseline_args=()
  if [[ -f "${GSM8K_SMALL_BASELINE}" ]]; then
    baseline_args=(--small-baseline-path "${GSM8K_SMALL_BASELINE}")
  else
    echo "[gsm8k] WARN: small baseline missing, continuing without override: ${GSM8K_SMALL_BASELINE}"
  fi

  echo "[gsm8k] scheduler sweep -> ${GSM8K_TRACE}"
  python -m core_package.schedulers.simulate_observe_rollback_scheduler \
    --label-path "${GSM8K_LABEL}" \
    --eval-data-path "${GSM8K_EVAL_PT}" \
    --probe-artifact-path "${GSM8K_PROBE}" \
    "${baseline_args[@]}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH}" \
    --large-backend hf \
    --thresholds "${THRESHOLDS}" \
    --num-test-questions 300 \
    --max-new-tokens "${GSM8K_MAX_NEW_TOKENS}" \
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
    --trace-export-path "${GSM8K_TRACE}" \
    2>&1 | tee "${LOG_DIR}/03_schedule_gsm8k_clean.log"
}

run_svamp() {
  echo "========== SVAMP clean rerun =========="
  test -f "${SVAMP_LABEL}"
  test -f "${SVAMP_TEST_JSONL}"

  if [[ "${SKIP_BUILD_EVAL}" != "1" ]]; then
    echo "[svamp] build test-only hidden states -> ${SVAMP_EVAL_PT}"
    python -m core_package.pipelines.build_dataset \
      --dataset-name svamp \
      --input-path "${SVAMP_TEST_JSONL}" \
      --answer-type svamp_boxed_numeric \
      --num-samples 300 \
      --model-path "${SMALL_MODEL_PATH}" \
      --save-path "${SVAMP_EVAL_PT}" \
      --max-new-tokens "${SVAMP_MAX_NEW_TOKENS}" \
      2>&1 | tee "${LOG_DIR}/04_build_svamp_test_hidden.log"
  fi

  check_no_overlap "${SVAMP_LABEL}" "${SVAMP_EVAL_PT}" "svamp"

  if [[ "${SKIP_TRAIN_PROBE}" != "1" ]]; then
    echo "[svamp] train probe artifact -> ${SVAMP_PROBE}"
    python -m core_package.probes.train_probe_artifact_torch \
      --label-path "${SVAMP_LABEL}" \
      --output-path "${SVAMP_PROBE}" \
      2>&1 | tee "${LOG_DIR}/05_train_svamp_probe.log"
  fi

  local baseline_args=()
  if [[ -f "${SVAMP_SMALL_BASELINE}" ]]; then
    baseline_args=(--small-baseline-path "${SVAMP_SMALL_BASELINE}")
  else
    echo "[svamp] WARN: small baseline missing, continuing without override: ${SVAMP_SMALL_BASELINE}"
  fi

  echo "[svamp] scheduler sweep -> ${SVAMP_TRACE}"
  python -m core_package.schedulers.simulate_observe_rollback_scheduler_svamp \
    --label-path "${SVAMP_LABEL}" \
    --eval-data-path "${SVAMP_EVAL_PT}" \
    --probe-artifact-path "${SVAMP_PROBE}" \
    "${baseline_args[@]}" \
    --small-model-path "${SMALL_MODEL_PATH}" \
    --large-model-path "${LARGE_MODEL_PATH}" \
    --large-backend hf \
    --thresholds "${THRESHOLDS}" \
    --num-test-questions 300 \
    --max-new-tokens "${SVAMP_MAX_NEW_TOKENS}" \
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
    --trace-export-path "${SVAMP_TRACE}" \
    2>&1 | tee "${LOG_DIR}/06_schedule_svamp_clean.log"
}

if [[ "${RUN_GSM8K}" == "1" ]]; then
  run_gsm8k
fi
if [[ "${RUN_SVAMP}" == "1" ]]; then
  run_svamp
fi

echo "[done] clean rerun finished"
echo "[done] logs: ${LOG_DIR}"
echo "[done] GSM8K trace: ${GSM8K_TRACE}"
echo "[done] SVAMP trace: ${SVAMP_TRACE}"
