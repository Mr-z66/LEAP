#!/usr/bin/env bash
set -euo pipefail

cd ~/care_experiment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate care_env

PROMPT='You are a helpful math assistant. Please reason step by step, and put your final answer within \boxed{}.'

echo "[1/3] build_dataset start: $(date)"
python -m core_package.pipelines.build_dataset \
  --dataset-name math500 \
  --input-path dataset/math500/test.jsonl \
  --num-samples 500 \
  --model-path models/Qwen2.5-1.5B \
  --save-path dataset/math500_test_15b_hidden_states_boxed.pt \
  --max-new-tokens 768 \
  --answer-type boxed \
  --system-prompt "$PROMPT"
echo "[1/3] build_dataset done: $(date)"

echo "[2/3] 1.5B eval start: $(date)"
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/math500_test_15b_hidden_states_boxed.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-1.5B \
  --max-new-tokens 768 \
  --answer-type boxed \
  --system-prompt "$PROMPT" \
  --output-path result/analysis_outputs/qwen25_15b_only_math500_boxed.json
echo "[2/3] 1.5B eval done: $(date)"

echo "[3/3] 32B eval start: $(date)"
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/math500_test_15b_hidden_states_boxed.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-32B \
  --max-new-tokens 768 \
  --answer-type boxed \
  --system-prompt "$PROMPT" \
  --output-path result/analysis_outputs/qwen25_32b_only_math500_boxed.json
echo "[3/3] 32B eval done: $(date)"

echo "Stage 1 finished successfully: $(date)"
