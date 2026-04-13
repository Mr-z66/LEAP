#!/usr/bin/env bash
set -euo pipefail

cd ~/care_experiment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate care_env

echo "[1/3] build_dataset start: $(date)"
python -m core_package.pipelines.build_dataset \
  --dataset-name math500 \
  --input-path dataset/math500/test.jsonl \
  --num-samples 500 \
  --model-path models/Qwen2.5-Math-1.5B-Instruct \
  --save-path dataset/math500_test_15b_hidden_states_qwen_math.pt \
  --max-new-tokens 768
echo "[1/3] build_dataset done: $(date)"

echo "[2/3] 1.5B eval start: $(date)"
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/math500_test_15b_hidden_states_qwen_math.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-Math-1.5B-Instruct \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_math_15b_only_math500.json
echo "[2/3] 1.5B eval done: $(date)"

echo "[3/3] 32B eval start: $(date)"
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/math500_test_15b_hidden_states_qwen_math.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-Math-32B-Instruct \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_math_32b_only_math500.json
echo "[3/3] 32B eval done: $(date)"

echo "Stage 1 finished successfully: $(date)"
