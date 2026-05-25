#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/LEAP}"
CONDA_ENV="${CONDA_ENV:-care_env}"
LABEL_DIR="${LABEL_DIR:-dataset/mixed_probe_labels_fallback_second_pass}"
ARTIFACT_PATH="${ARTIFACT_PATH:-result/artifacts/probe_artifact_mixed_gsm8k_svamp.pt}"
DATASETS="${DATASETS:-gsm8k_test svamp_test}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"

cd "${ROOT_DIR}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p result/logs/mixed_probe_mainline

for key in ${DATASETS}; do
  data_path="${LABEL_DIR}/${key}_labels.pt"
  log_path="result/logs/mixed_probe_mainline/probe_eval_${key}_$(date +%Y%m%d_%H%M%S).log"
  if [[ ! -f "${data_path}" ]]; then
    echo "[skip] missing labeled eval file: ${data_path}"
    continue
  fi

  echo
  echo "========== probe eval ${key} =========="
  echo "[data] ${data_path}"
  echo "[artifact] ${ARTIFACT_PATH}"
  echo "[log] ${log_path}"

  python -m core_package.probes.evaluate_probe_baseline_torch \
    --data-path "${data_path}" \
    --artifact-path "${ARTIFACT_PATH}" \
    --threshold-grid "${THRESHOLD_GRID}" \
    --all-questions \
    2>&1 | tee "${log_path}"
done

echo
echo "[done] mixed probe eval finished"
