#!/usr/bin/env bash
set -euo pipefail

cd ~/care_experiment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate care_env

# -----------------------------
# MATH500 overnight HF pipeline
# small: Qwen2.5-Math-1.5B-Instruct
# large: Qwen2.5-32B
# backend: Hugging Face (no vLLM)
# -----------------------------

SMALL_MODEL_PATH="models/Qwen2.5-Math-1.5B-Instruct"
LARGE_MODEL_PATH="models/Qwen2.5-32B"

INPUT_JSONL="dataset/math500/test.jsonl"
BUILD_OUTPUT="dataset/math500_test_15b_hidden_states_hf_t2048.pt"
LABEL_OUTPUT="dataset/math500_labeled_data_strict_hf_t2048.pt"

SMALL_ONLY_JSON="result/analysis_outputs/qwen25_math_15b_only_math500_hf_t2048.json"
LARGE_ONLY_JSON="result/analysis_outputs/qwen25_32b_only_math500_hf_t2048.json"
PROBE_ARTIFACT="result/artifacts/probe_artifact_math500_hf_t2048.pt"
SCHED_TRACE_JSON="result/traces/observe_rollback_traces_math500_hf_plain32b_t2048.json"

NUM_SAMPLES=500
MAX_NEW_TOKENS=2048
THRESHOLDS="0.20,0.25,0.30,0.35,0.40"
ANSWER_TYPE="math500_qwen_boxed"
FEATURE_KEY="boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob"

mkdir -p dataset result result/analysis_outputs result/artifacts result/traces

log_step() {
  echo
  echo "=================================================="
  echo "$1"
  echo "time: $(date '+%F %T')"
  echo "=================================================="
}

run_or_skip() {
  local target="$1"
  shift
  if [[ -f "$target" ]]; then
    echo "[skip] found existing file: $target"
  else
    "$@"
  fi
}

log_step "[1/6] Build small-model trajectories"
run_or_skip "$BUILD_OUTPUT" \
  python -m core_package.pipelines.build_dataset \
    --dataset-name math500 \
    --input-path "$INPUT_JSONL" \
    --num-samples "$NUM_SAMPLES" \
    --model-path "$SMALL_MODEL_PATH" \
    --save-path "$BUILD_OUTPUT" \
    --max-new-tokens "$MAX_NEW_TOKENS"

log_step "[2/6] Strict labeling with large model"
if [[ -f "$LABEL_OUTPUT" ]]; then
  python -m core_package.pipelines.referee_32b_labeling_strict \
    --input-path "$BUILD_OUTPUT" \
    --output-path "$LABEL_OUTPUT" \
    --model-path "$LARGE_MODEL_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --save-every 10 \
    --resume \
    --include-reference-answer
else
  python -m core_package.pipelines.referee_32b_labeling_strict \
    --input-path "$BUILD_OUTPUT" \
    --output-path "$LABEL_OUTPUT" \
    --model-path "$LARGE_MODEL_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --save-every 10 \
    --include-reference-answer
fi

log_step "[3/6] Small-model only evaluation"
run_or_skip "$SMALL_ONLY_JSON" \
  python -m evaluation.evaluate_model_only_accuracy \
    --label-path "$BUILD_OUTPUT" \
    --artifact-path does_not_exist.pt \
    --trace-path does_not_exist.json \
    --model-path "$SMALL_MODEL_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --answer-type "$ANSWER_TYPE" \
    --output-path "$SMALL_ONLY_JSON"

log_step "[4/6] Large-model only evaluation"
run_or_skip "$LARGE_ONLY_JSON" \
  python -m evaluation.evaluate_model_only_accuracy \
    --label-path "$BUILD_OUTPUT" \
    --artifact-path does_not_exist.pt \
    --trace-path does_not_exist.json \
    --model-path "$LARGE_MODEL_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --answer-type "$ANSWER_TYPE" \
    --output-path "$LARGE_ONLY_JSON"

log_step "[5/6] Train probe"
run_or_skip "$PROBE_ARTIFACT" \
  python -m core_package.probes.train_probe_artifact_torch \
    --label-path "$LABEL_OUTPUT" \
    --output-path "$PROBE_ARTIFACT" \
    --feature-key "$FEATURE_KEY" \
    --hidden-layers 128,32 \
    --dropout 0.1 \
    --epochs 60 \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --weight-decay 1e-3 \
    --low-entropy-error-final-entropy-max 1.0 \
    --low-entropy-error-final-top1-min 0.9 \
    --low-entropy-error-weight 4.0

log_step "[6/6] Observe-and-rollback scheduler"
python -m core_package.schedulers.simulate_observe_rollback_scheduler \
  --label-path "$LABEL_OUTPUT" \
  --eval-data-path "$BUILD_OUTPUT" \
  --probe-artifact-path "$PROBE_ARTIFACT" \
  --small-baseline-path "$SMALL_ONLY_JSON" \
  --small-model-path "$SMALL_MODEL_PATH" \
  --large-model-path "$LARGE_MODEL_PATH" \
  --thresholds "$THRESHOLDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-handoffs 2 \
  --large-handoff-chunks 2 \
  --cooldown-chunks 2 \
  --answer-type "$ANSWER_TYPE" \
  --trace-export-path "$SCHED_TRACE_JSON"

log_step "MATH500 HF pipeline finished"
echo "build_output      = $BUILD_OUTPUT"
echo "label_output      = $LABEL_OUTPUT"
echo "small_only_json   = $SMALL_ONLY_JSON"
echo "large_only_json   = $LARGE_ONLY_JSON"
echo "probe_artifact    = $PROBE_ARTIFACT"
echo "scheduler_trace   = $SCHED_TRACE_JSON"
