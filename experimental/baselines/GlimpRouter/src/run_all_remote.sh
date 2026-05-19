#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/result/baselines/glimprouter_qwen15b_qwen32b_${STAMP}}"

export GLIMP_API_KEY="${GLIMP_API_KEY:-token-abc123}"
export GLIMP_MODEL_32B="${GLIMP_MODEL_32B:-Qwen2.5-32B}"
export GLIMP_MODEL_1P5B="${GLIMP_MODEL_1P5B:-Qwen2.5-Math-1.5B-Instruct}"
export GLIMP_BASE_URL_32B="${GLIMP_BASE_URL_32B:-http://127.0.0.1:8000/v1}"
export GLIMP_BASE_URL_1P5B="${GLIMP_BASE_URL_1P5B:-http://127.0.0.1:8001/v1}"

mkdir -p "$OUTPUT_ROOT"

echo "Running GlimpRouter benchmark suite"
echo "ROOT_DIR=$ROOT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "GLIMP_BASE_URL_32B=$GLIMP_BASE_URL_32B"
echo "GLIMP_BASE_URL_1P5B=$GLIMP_BASE_URL_1P5B"

EXTRA_ARGS=()
if [[ -n "${GLIMP_MAX_QUESTIONS:-}" ]]; then
  EXTRA_ARGS+=(--max-questions "$GLIMP_MAX_QUESTIONS")
fi

python -u experimental/baselines/GlimpRouter/src/run_leap_benchmarks.py \
  --datasets gsm8k,svamp,math500 \
  --output-root "$OUTPUT_ROOT" \
  --repeat-num 1 \
  --score-method first_token_entropy \
  --score-threshold "${GLIMP_SCORE_THRESHOLD:-1.0}" \
  --model-size 32b \
  --small-model-size 1.5b \
  --gsm8k-token-budget "${GLIMP_GSM8K_TOKEN_BUDGET:-2048}" \
  --svamp-token-budget "${GLIMP_SVAMP_TOKEN_BUDGET:-2048}" \
  --math500-token-budget "${GLIMP_MATH500_TOKEN_BUDGET:-4096}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/run.log"

echo
echo "Finished. Summaries:"
echo "  $OUTPUT_ROOT/glimprouter_benchmark_summary.json"
echo "  $OUTPUT_ROOT/glimprouter_benchmark_summary.csv"
