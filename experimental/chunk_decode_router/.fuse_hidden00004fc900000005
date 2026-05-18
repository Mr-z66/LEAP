#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/care_experiment}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen2.5-32B}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-32B}"
API_KEY="${API_KEY:-token-abc123}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV_NAME:-care_env}"

echo "[chunk_decode_router] starting vLLM 32B server"
echo "  project_root=${PROJECT_ROOT}"
echo "  model_path=${MODEL_PATH}"
echo "  cuda_devices=${CUDA_DEVICES}"
echo "  served_model_name=${SERVED_MODEL_NAME}"
echo "  port=${PORT}"
echo "  tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
echo "  gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "  max_model_len=${MAX_MODEL_LEN}"
echo "  enforce_eager=${ENFORCE_EAGER}"

if [ "${ENFORCE_EAGER}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --enforce-eager
else
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}"
fi
