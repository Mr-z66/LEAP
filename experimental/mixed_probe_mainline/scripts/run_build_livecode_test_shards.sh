#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
INPUT_PATH="${INPUT_PATH:-dataset/mixed_probe_splits/livecodebench_v5_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/mixed_probe_trajectories_fallback_shards}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_DIR}" result/logs/mixed_probe_mainline

build_shard() {
  local start="$1"
  local end="$2"
  local count=$((end - start))
  local save_path
  local log_path
  save_path=$(printf "%s/livecodebench_v5_test_%03d_%03d_15b.pt" "${OUTPUT_DIR}" "${start}" "${end}")
  log_path=$(printf "result/logs/mixed_probe_mainline/build_livecodebench_v5_test_%03d_%03d.log" "${start}" "${end}")

  echo
  echo "========== build livecodebench_v5/test | ${start}:${end} =========="
  echo "[input] ${INPUT_PATH}"
  echo "[save]  ${save_path}"
  echo "[log]   ${log_path}"

  python -m core_package.pipelines.build_dataset \
    --dataset-name livecodebench_v5 \
    --input-path "${INPUT_PATH}" \
    --answer-type livecodebench_codegen \
    --start-question "${start}" \
    --num-samples "${count}" \
    --model-path "${SMALL_MODEL_PATH}" \
    --save-path "${save_path}" \
    --chunking-method code_rsd_fallback \
    --step-word $'\n\n' \
    --min-step-tokens 12 \
    --target-step-tokens 96 \
    --max-step-tokens 160 \
    --force-step-tokens 256 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${log_path}"
}

build_shard 0 10
build_shard 10 20
build_shard 20 30
build_shard 30 40
build_shard 40 50
build_shard 50 60
build_shard 60 67

echo
echo "[done] livecodebench_v5 test shards saved under ${OUTPUT_DIR}"
