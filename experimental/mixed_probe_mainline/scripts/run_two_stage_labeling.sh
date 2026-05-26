#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
TRAJ_DIR="${TRAJ_DIR:-dataset/mixed_probe_trajectories_fallback}"
STAGE1_DIR="${STAGE1_DIR:-dataset/mixed_probe_labels_fallback_stage1}"
STAGE2_DIR="${STAGE2_DIR:-dataset/mixed_probe_labels_fallback_second_pass}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-32B}"
JUDGE_BACKEND="${JUDGE_BACKEND:-hf}"
MODE="${MODE:-smoke}" # smoke or full
DATASETS="${DATASETS:-gsm8k_calib gsm8k_test svamp_calib svamp_test math500_calib math500_test}"

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
  NUM_SAMPLES_ARG=(--num-samples "${NUM_SAMPLES:-5}")
  MAX_QUESTIONS_ARG=(--max-questions "${MAX_QUESTIONS:-5}")
elif [[ "${MODE}" == "full" ]]; then
  NUM_SAMPLES_ARG=()
  MAX_QUESTIONS_ARG=()
else
  echo "MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

mkdir -p "${STAGE1_DIR}" "${STAGE2_DIR}" result/logs/mixed_probe_mainline

trajectory_path_for() {
  case "$1" in
    gsm8k_calib) echo "${TRAJ_DIR}/gsm8k_calib_300_15b.pt" ;;
    gsm8k_test) echo "${TRAJ_DIR}/gsm8k_test_300_15b.pt" ;;
    svamp_calib) echo "${TRAJ_DIR}/svamp_calib_300_15b.pt" ;;
    svamp_test) echo "${TRAJ_DIR}/svamp_test_300_15b.pt" ;;
    math500_calib) echo "${TRAJ_DIR}/math500_calib_300_15b.pt" ;;
    math500_test) echo "${TRAJ_DIR}/math500_test_300_15b.pt" ;;
    livecodebench_v5_calib) echo "${TRAJ_DIR}/livecodebench_v5_calib_100_15b.pt" ;;
    livecodebench_v5_test) echo "${TRAJ_DIR}/livecodebench_v5_test_67_15b.pt" ;;
    *)
      echo "Unknown dataset key: $1" >&2
      return 1
      ;;
  esac
}

run_stage1() {
  local key="$1"
  local input_path="$2"
  local output_path="${STAGE1_DIR}/${key}_labels.pt"
  local log_path="result/logs/mixed_probe_mainline/label_stage1_${key}_${MODE}.log"

  echo
  echo "========== stage1 label ${key} =========="
  echo "[input]  ${input_path}"
  echo "[output] ${output_path}"
  echo "[log]    ${log_path}"

  python -m core_package.pipelines.label_existing_trajectories \
    --input-path "${input_path}" \
    --output-path "${output_path}" \
    "${NUM_SAMPLES_ARG[@]}" \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --judge-backend "${JUDGE_BACKEND}" \
    --max-judge-tokens 192 \
    --lookahead-steps 1 \
    --save-every 5 \
    --resume \
    2>&1 | tee "${log_path}"
}

run_stage2() {
  local key="$1"
  local input_path="${STAGE1_DIR}/${key}_labels.pt"
  local output_path="${STAGE2_DIR}/${key}_labels.pt"
  local log_path="result/logs/mixed_probe_mainline/label_stage2_${key}_${MODE}.log"

  echo
  echo "========== stage2 refine ${key} =========="
  echo "[input]  ${input_path}"
  echo "[output] ${output_path}"
  echo "[log]    ${log_path}"

  python -m core_package.pipelines.refine_clean_step_labels_second_pass \
    --input-path "${input_path}" \
    --output-path "${output_path}" \
    "${MAX_QUESTIONS_ARG[@]}" \
    --judge-model-path "${JUDGE_MODEL_PATH}" \
    --judge-backend "${JUDGE_BACKEND}" \
    --max-judge-tokens 384 \
    --low-confidence-threshold 0.55 \
    2>&1 | tee "${log_path}"
}

for key in ${DATASETS}; do
  input_path="$(trajectory_path_for "${key}")"
  if [[ ! -f "${input_path}" ]]; then
    echo "[skip] missing trajectory: ${input_path}"
    continue
  fi
  run_stage1 "${key}" "${input_path}"
  run_stage2 "${key}"
done

echo
echo "[done] stage1 labels: ${STAGE1_DIR}"
echo "[done] stage2 labels: ${STAGE2_DIR}"
