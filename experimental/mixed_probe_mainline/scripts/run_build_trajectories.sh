#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
MODE="${MODE:-smoke}" # smoke or full
OUTPUT_DIR="${OUTPUT_DIR:-dataset/mixed_probe_trajectories}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "${MODE}" == "smoke" ]]; then
  GSM8K_N="${GSM8K_N:-2}"
  SVAMP_N="${SVAMP_N:-2}"
  MATH500_N="${MATH500_N:-2}"
  LCB_N="${LCB_N:-2}"
elif [[ "${MODE}" == "full" ]]; then
  GSM8K_N="${GSM8K_N:-300}"
  SVAMP_N="${SVAMP_N:-300}"
  MATH500_N="${MATH500_N:-300}"
  LCB_N="${LCB_N:-100}"
else
  echo "MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" result/logs/mixed_probe_mainline

run_build() {
  local dataset_name="$1"
  local split_name="$2"
  local input_path="$3"
  local answer_type="$4"
  local num_samples="$5"
  local max_new_tokens="$6"
  local save_path="${OUTPUT_DIR}/${dataset_name}_${split_name}_${num_samples}_15b.pt"
  local log_path="result/logs/mixed_probe_mainline/build_${dataset_name}_${split_name}_${num_samples}_${MODE}.log"

  echo
  echo "========== build ${dataset_name}/${split_name} | n=${num_samples} =========="
  echo "[input] ${input_path}"
  echo "[save]  ${save_path}"
  echo "[log]   ${log_path}"

  python -m core_package.pipelines.build_dataset \
    --dataset-name "${dataset_name}" \
    --input-path "${input_path}" \
    --answer-type "${answer_type}" \
    --num-samples "${num_samples}" \
    --model-path "${SMALL_MODEL_PATH}" \
    --save-path "${save_path}" \
    --chunking-method rsd_step_fallback \
    --step-word $'\n\n' \
    --min-step-tokens 12 \
    --target-step-tokens 64 \
    --max-step-tokens 120 \
    --force-step-tokens 180 \
    --max-new-tokens "${max_new_tokens}" \
    2>&1 | tee "${log_path}"
}

run_build gsm8k calib dataset/mixed_probe_splits/gsm8k_calib.jsonl gsm8k_boxed_numeric "${GSM8K_N}" 768
run_build gsm8k test dataset/mixed_probe_splits/gsm8k_test.jsonl gsm8k_boxed_numeric "${GSM8K_N}" 768

run_build svamp calib dataset/mixed_probe_splits/svamp_calib.jsonl svamp_boxed_numeric "${SVAMP_N}" 512
run_build svamp test dataset/mixed_probe_splits/svamp_test.jsonl svamp_boxed_numeric "${SVAMP_N}" 512

run_build math500 calib dataset/mixed_probe_splits/math500_calib.jsonl math500_qwen_boxed "${MATH500_N}" 1024
run_build math500 test dataset/mixed_probe_splits/math500_test.jsonl math500_qwen_boxed "${MATH500_N}" 1024

run_build livecodebench_v5 calib dataset/mixed_probe_splits/livecodebench_v5_calib.jsonl livecodebench_codegen "${LCB_N}" 1536

if [[ "${MODE}" == "full" ]]; then
  run_build livecodebench_v5 test dataset/mixed_probe_splits/livecodebench_v5_test.jsonl livecodebench_codegen "${LCB_TEST_N:-67}" 1536
else
  run_build livecodebench_v5 test dataset/mixed_probe_splits/livecodebench_v5_test.jsonl livecodebench_codegen "${LCB_N}" 1536
fi

echo
echo "[done] trajectories saved under ${OUTPUT_DIR}"
