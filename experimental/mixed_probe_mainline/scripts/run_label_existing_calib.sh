#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
TRAJ_DIR="${TRAJ_DIR:-dataset/mixed_probe_trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/mixed_probe_labels_existing}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-32B}"
JUDGE_BACKEND="${JUDGE_BACKEND:-hf}"
MODE="${MODE:-smoke}" # smoke or full
DATASETS="${DATASETS:-gsm8k_calib svamp_calib math500_calib livecodebench_v5_calib}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${MODE}" == "smoke" ]]; then
  GSM8K_N="${GSM8K_N:-5}"
  SVAMP_N="${SVAMP_N:-5}"
  MATH500_N="${MATH500_N:-5}"
  LCB_N="${LCB_N:-3}"
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

run_label() {
  local dataset_name="$1"
  local input_path="$2"
  local output_path="$3"
  local num_samples="$4"
  local log_path="result/logs/mixed_probe_mainline/label_existing_${dataset_name}_${num_samples}_${MODE}.log"

  echo
  echo "========== label existing ${dataset_name} | n=${num_samples} =========="
  echo "[input]  ${input_path}"
  echo "[output] ${output_path}"
  echo "[judge]  ${JUDGE_MODEL_PATH} (${JUDGE_BACKEND})"
  echo "[log]    ${log_path}"

  [[ -f "${input_path}" ]] || { echo "[error] missing trajectory: ${input_path}" >&2; exit 2; }

  python -m core_package.pipelines.label_existing_trajectories \
    --input-path "${input_path}" \
    --output-path "${output_path}" \
    --num-samples "${num_samples}" \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --judge-backend "${JUDGE_BACKEND}" \
    --max-judge-tokens 192 \
    --save-every 5 \
    --resume \
    2>&1 | tee "${log_path}"
}

for key in ${DATASETS}; do
  case "${key}" in
    gsm8k_calib) run_label gsm8k "${TRAJ_DIR}/gsm8k_calib_300_15b.pt" "${OUTPUT_DIR}/gsm8k_calib_labels.pt" "${GSM8K_N}" ;;
    svamp_calib) run_label svamp "${TRAJ_DIR}/svamp_calib_300_15b.pt" "${OUTPUT_DIR}/svamp_calib_labels.pt" "${SVAMP_N}" ;;
    math500_calib) run_label math500 "${TRAJ_DIR}/math500_calib_300_15b.pt" "${OUTPUT_DIR}/math500_calib_labels.pt" "${MATH500_N}" ;;
    livecodebench_v5_calib) run_label livecodebench_v5 "${TRAJ_DIR}/livecodebench_v5_calib_100_15b.pt" "${OUTPUT_DIR}/livecodebench_v5_calib_labels.pt" "${LCB_N}" ;;
    *) echo "[error] unsupported dataset key: ${key}" >&2; exit 2 ;;
  esac
done

echo
echo "[done] labels saved under ${OUTPUT_DIR}"
