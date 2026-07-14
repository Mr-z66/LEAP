#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/LEAP}"
DATA="${DATA:-/root/autodl-tmp/LEAP_data}"
SMALL="${SMALL:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE="${LARGE:-/root/autodl-tmp/models/Qwen2.5-7B}"
LABELS="$DATA/dataset/commonsense_labels"
ART="$DATA/result/artifacts/commonsense"
OUT="$ROOT/result/baselines/commonsense"
LOG="$ROOT/result/logs/commonsense"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate care_env
cd "$ROOT"
mkdir -p "$OUT" "$LOG"

wait_for_gpu_headroom() {
  local required_free_mb="${1:-24000}"
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= required_free_mb )); then
      echo "[only-baselines] GPU headroom ready: free=${free_mb}MiB required=${required_free_mb}MiB"
      return 0
    fi
    echo "[only-baselines] waiting for GPU headroom: free=${free_mb:-unknown}MiB required=${required_free_mb}MiB $(date -Is)"
    sleep 30
  done
}

run_one() {
  local dataset="$1" model_tag="$2" model_path="$3" params_b="$4"
  local label_path="$DATA/dataset/commonsense_trajectories/${dataset}_calib_holdout.pt"
  local output_path="$OUT/${dataset}_${model_tag}_only.json"
  local log_path="$LOG/${dataset}_${model_tag}_only.log"

  echo "[only-baselines] start dataset=$dataset model=$model_tag $(date -Is)"
  python -m evaluation.evaluate_model_only_accuracy \
    --label-path "$label_path" \
    --artifact-path "" \
    --trace-path "" \
    --model-path "$model_path" \
    --model-params-b "$params_b" \
    --answer-type multiple_choice_letter \
    --max-new-tokens 256 \
    --num-test-questions 60 \
    --output-path "$output_path" \
    > "$log_path" 2>&1
  echo "[only-baselines] done dataset=$dataset model=$model_tag $(date -Is)"
}

wait_for_gpu_headroom 24000
run_one commonsenseqa 1.5b "$SMALL" 1.5
if [[ "${ONLY_CS_SMALL:-0}" == "1" ]]; then
  echo "[only-baselines] requested CommonsenseQA 1.5B-only run complete $(date -Is)"
  exit 0
fi
wait_for_gpu_headroom 24000
run_one commonsenseqa 7b "$LARGE" 7.0
wait_for_gpu_headroom 24000
run_one arc_challenge 1.5b "$SMALL" 1.5
wait_for_gpu_headroom 24000
run_one arc_challenge 7b "$LARGE" 7.0
echo "[only-baselines] all complete $(date -Is)"
