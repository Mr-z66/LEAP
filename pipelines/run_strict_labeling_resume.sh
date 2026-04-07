#!/usr/bin/env bash
set -euo pipefail

TARGET_SAMPLES="${TARGET_SAMPLES:-500}"
INPUT_PATH="${INPUT_PATH:-gsm8k_15b_hidden_states.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-gsm8k_labeled_training_data_strict.pt}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-32B}"
MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-64}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
INCLUDE_REFERENCE_ANSWER="${INCLUDE_REFERENCE_ANSWER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

count_progress() {
  python "${SCRIPT_DIR}/count_labeled_questions.py" --output-path "${OUTPUT_PATH}"
}

echo "[strict-resume] repo_root=${REPO_ROOT}"
echo "[strict-resume] target_samples=${TARGET_SAMPLES}"
echo "[strict-resume] input=${INPUT_PATH}"
echo "[strict-resume] output=${OUTPUT_PATH}"
echo "[strict-resume] model=${MODEL_PATH}"
echo "[strict-resume] max_judge_tokens=${MAX_JUDGE_TOKENS}"

while true; do
  PROGRESS="$(count_progress)"
  CURRENT_QUESTIONS="$(echo "${PROGRESS}" | sed -E 's/.*questions=([0-9]+).*/\1/')"
  CURRENT_CHUNKS="$(echo "${PROGRESS}" | sed -E 's/.*chunks=([0-9]+).*/\1/')"

  echo "[strict-resume] $(date '+%F %T') progress: questions=${CURRENT_QUESTIONS} chunks=${CURRENT_CHUNKS}"

  if [[ "${CURRENT_QUESTIONS}" -ge "${TARGET_SAMPLES}" ]]; then
    echo "[strict-resume] target reached, stop."
    exit 0
  fi

  CMD=(
    python "${SCRIPT_DIR}/referee_32b_labeling_strict.py"
    --input-path "${INPUT_PATH}"
    --output-path "${OUTPUT_PATH}"
    --model-path "${MODEL_PATH}"
    --num-samples "${TARGET_SAMPLES}"
    --max-judge-tokens "${MAX_JUDGE_TOKENS}"
    --save-every "${SAVE_EVERY}"
    --resume
    --stop-after-first-error
  )

  if [[ "${INCLUDE_REFERENCE_ANSWER}" == "1" ]]; then
    CMD+=(--include-reference-answer)
  fi

  echo "[strict-resume] launching strict labeling worker..."
  (
    cd "${REPO_ROOT}"
    "${CMD[@]}"
  )

  echo "[strict-resume] worker exited, rechecking progress after ${SLEEP_SECONDS}s..."
  sleep "${SLEEP_SECONDS}"
done
