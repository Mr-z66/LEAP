#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-result/logs/boxed_strict_labeling_${TIMESTAMP}}"
BACKUP_DIR="${BACKUP_DIR:-result/backups/boxed_strict_labeling_${TIMESTAMP}}"

SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-models/Qwen2.5-1.5B}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/Qwen2.5-32B}"
BACKEND="${BACKEND:-hf}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen2.5-32B}"

GSM8K_TARGET="${GSM8K_TARGET:-300}"
SVAMP_TARGET="${SVAMP_TARGET:-200}"
SAVE_EVERY="${SAVE_EVERY:-5}"

GSM8K_TRAJ="${GSM8K_TRAJ:-dataset/gsm8k_15b_hidden_states_boxed.pt}"
GSM8K_LABELS="${GSM8K_LABELS:-dataset/gsm8k_labeled_training_data_strict_boxed.pt}"
SVAMP_TRAJ="${SVAMP_TRAJ:-dataset/svamp_15b_hidden_states_boxed.pt}"
SVAMP_LABELS="${SVAMP_LABELS:-dataset/svamp_labeled_training_data_strict_boxed.pt}"

mkdir -p "${LOG_DIR}" "${BACKUP_DIR}" dataset result/logs result/backups

backup_if_exists() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    local base
    base="$(basename "${path}")"
    cp -p "${path}" "${BACKUP_DIR}/${base}.${TIMESTAMP}.bak"
    echo "[backup] ${path} -> ${BACKUP_DIR}/${base}.${TIMESTAMP}.bak"
  fi
}

judge_backend_args=()
if [[ "${BACKEND}" == "vllm" ]]; then
  judge_backend_args=(
    --backend vllm
    --vllm-base-url "${VLLM_BASE_URL}"
    --vllm-api-key "${VLLM_API_KEY}"
    --vllm-model-name "${VLLM_MODEL_NAME}"
  )
else
  judge_backend_args=(--backend hf)
fi

echo "[run] root=${ROOT_DIR}"
echo "[run] logs=${LOG_DIR}"
echo "[run] backups=${BACKUP_DIR}"
echo "[run] backend=${BACKEND}"
echo "[run] judge_model=${JUDGE_MODEL_PATH}"
echo "[run] targets: gsm8k=${GSM8K_TARGET}, svamp=${SVAMP_TARGET}"

backup_if_exists "${GSM8K_LABELS}"
backup_if_exists "${SVAMP_LABELS}"

if [[ ! -f "${GSM8K_TRAJ}" ]]; then
  echo "[build] missing ${GSM8K_TRAJ}; building GSM8K boxed trajectories"
  python -m core_package.pipelines.build_dataset \
    --dataset-name gsm8k \
    --dataset-split "train[:${GSM8K_TARGET}]" \
    --model-path "${SMALL_MODEL_PATH}" \
    --save-path "${GSM8K_TRAJ}" \
    --answer-type gsm8k_boxed_numeric \
    --max-new-tokens 768 \
    2>&1 | tee "${LOG_DIR}/01_build_gsm8k_boxed.log"
fi

echo "[label] GSM8K boxed strict labels -> ${GSM8K_LABELS}"
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path "${GSM8K_TRAJ}" \
  --output-path "${GSM8K_LABELS}" \
  --model-path "${JUDGE_MODEL_PATH}" \
  --num-samples "${GSM8K_TARGET}" \
  --save-every "${SAVE_EVERY}" \
  --include-reference-answer \
  --resume \
  "${judge_backend_args[@]}" \
  2>&1 | tee "${LOG_DIR}/02_label_gsm8k_boxed.log"

python -m core_package.pipelines.count_labeled_questions \
  --output-path "${GSM8K_LABELS}" \
  2>&1 | tee "${LOG_DIR}/03_count_gsm8k_boxed.log"

if [[ ! -f "${SVAMP_TRAJ}" ]]; then
  echo "[build] missing ${SVAMP_TRAJ}; building SVAMP boxed trajectories"
  python -m core_package.pipelines.build_dataset \
    --dataset-name svamp \
    --input-path dataset/svamp/train.jsonl \
    --num-samples 700 \
    --model-path "${SMALL_MODEL_PATH}" \
    --save-path "${SVAMP_TRAJ}" \
    --answer-type svamp_boxed_numeric \
    --max-new-tokens 512 \
    2>&1 | tee "${LOG_DIR}/04_build_svamp_boxed.log"
fi

echo "[label] SVAMP boxed strict labels -> ${SVAMP_LABELS}"
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path "${SVAMP_TRAJ}" \
  --output-path "${SVAMP_LABELS}" \
  --model-path "${JUDGE_MODEL_PATH}" \
  --num-samples "${SVAMP_TARGET}" \
  --save-every "${SAVE_EVERY}" \
  --include-reference-answer \
  --resume \
  "${judge_backend_args[@]}" \
  2>&1 | tee "${LOG_DIR}/05_label_svamp_boxed.log"

python -m core_package.pipelines.count_labeled_questions \
  --output-path "${SVAMP_LABELS}" \
  2>&1 | tee "${LOG_DIR}/06_count_svamp_boxed.log"

echo "[done] boxed strict labeling finished"
echo "[done] logs: ${LOG_DIR}"
echo "[done] backups: ${BACKUP_DIR}"
