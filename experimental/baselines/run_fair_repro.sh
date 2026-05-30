#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"

DATASETS="${DATASETS:-gsm8k,svamp}"
MAX_QUESTIONS="${MAX_QUESTIONS:-300}"

GSM8K_PATH="${GSM8K_PATH:-dataset/mixed_probe_splits/gsm8k_test.jsonl}"
SVAMP_PATH="${SVAMP_PATH:-dataset/mixed_probe_splits/svamp_test.jsonl}"
MATH500_PATH="${MATH500_PATH:-dataset/mixed_probe_splits/math500_test.jsonl}"

SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE_MODEL_PATH="${LARGE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-/root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}"

GLIMP_THRESHOLDS="${GLIMP_THRESHOLDS:-0.6 0.8 1.0 1.2 1.4}"
RSD_THRESHOLDS="${RSD_THRESHOLDS:-0.6 0.7 0.8}"

RUN_GLIMP="${RUN_GLIMP:-1}"
RUN_RSD="${RUN_RSD:-1}"
RSD_BACKEND="${RSD_BACKEND:-hf}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GLIMP_HF_LOCAL_FILES_ONLY="${GLIMP_HF_LOCAL_FILES_ONLY:-1}"

mkdir -p result/baselines

if [[ "${RUN_GLIMP}" == "1" ]]; then
  for threshold in ${GLIMP_THRESHOLDS}; do
    tag="glimprouter_qwen15b_qwen7b_thr${threshold//./p}"
    python experimental/baselines/GlimpRouter/src/run_leap_benchmarks.py \
      --datasets "${DATASETS}" \
      --max-questions "${MAX_QUESTIONS}" \
      --small-backend hf \
      --large-backend hf \
      --small-model-size 1.5b \
      --model-size 7b \
      --small-model-path "${SMALL_MODEL_PATH}" \
      --large-model-path "${LARGE_MODEL_PATH}" \
      --score-method first_token_entropy \
      --score-threshold "${threshold}" \
      --gsm8k-path "${GSM8K_PATH}" \
      --svamp-path "${SVAMP_PATH}" \
      --math500-path "${MATH500_PATH}" \
      --gsm8k-token-budget 768 \
      --svamp-token-budget 512 \
      --math500-token-budget 1024 \
      --output-root "result/baselines/${tag}"
  done
fi

if [[ "${RUN_RSD}" == "1" ]]; then
  for threshold in ${RSD_THRESHOLDS}; do
    tag="rsd_qwen15b_qwen7b_prm${threshold//./p}_${RSD_BACKEND}"
    python experimental/baselines/RSD/run_leap_benchmarks.py \
      --backend "${RSD_BACKEND}" \
      --datasets "${DATASETS}" \
      --max-questions "${MAX_QUESTIONS}" \
      --draft-model-path "${SMALL_MODEL_PATH}" \
      --target-model-path "${LARGE_MODEL_PATH}" \
      --prm-model-path "${PRM_MODEL_PATH}" \
      --prm-threshold "${threshold}" \
      --max-tokens-per-call 768 \
      --gsm8k-path "${GSM8K_PATH}" \
      --svamp-path "${SVAMP_PATH}" \
      --math500-path "${MATH500_PATH}" \
      --draft-params-b 1.5 \
      --target-params-b 7.0 \
      --prm-params-b 1.5 \
      --output-root "result/baselines/${tag}"
  done
fi

python experimental/baselines/summarize_fair_repro.py \
  --roots result/baselines/glimprouter_qwen15b_qwen7b_thr* result/baselines/rsd_qwen15b_qwen7b_prm* \
  --output-csv result/baselines/fair_repro_summary.csv \
  --output-json result/baselines/fair_repro_summary.json

