#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/care_experiment}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-$PROJECT_ROOT/dataset/math500_test_15b_hidden_states_hf_t2048.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/experimental/chunk_decode_router/math500_test10_decode_choice_labeled.jsonl}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-32B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen2.5-32B}"
NUM_QUESTIONS="${NUM_QUESTIONS:-10}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/experimental/chunk_decode_router/math500_test${NUM_QUESTIONS}_decode_choice_labeled.jsonl}"

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV_NAME:-care_env}"

cd "${PROJECT_ROOT}"

python -m experimental.chunk_decode_router.build_decode_choice_dataset \
  --trajectory-path "${TRAJECTORY_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --dataset-name math500 \
  --answer-type math500_qwen_boxed \
  --num-questions "${NUM_QUESTIONS}" \
  --candidate-policy uniform_plus_ends \
  --candidate-count 4 \
  --label-with-rollouts \
  --small-model-path "${SMALL_MODEL_PATH}" \
  --large-model-path "${LARGE_MODEL_PATH}" \
  --large-backend vllm \
  --vllm-base-url "${VLLM_BASE_URL}" \
  --vllm-api-key "${VLLM_API_KEY}" \
  --vllm-model-name "${VLLM_MODEL_NAME}"
