#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python build_takeover_beneficial_labels.py \
  --small-model-path models/Qwen2.5-1.5B \
  --model-path models/Qwen2.5-32B \
  --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob \
  --candidate-mode hybrid \
  --top-k 2 \
  --explore-positions middle,last \
  --large-handoff-chunks 2 \
  --max-new-tokens 256 \
  --only-small-wrong \
  --num-questions 20

python train_probe_artifact.py \
  --label-path gsm8k_takeover_beneficial_labels.pt \
  --label-key takeover_beneficial \
  --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob \
  --output-path beneficial_probe_artifact.pt

python simulate_multi_handoff_scheduler.py \
  --probe-artifact-path beneficial_probe_artifact.pt \
  --thresholds 0.01,0.02,0.05,0.10,0.15,0.20 \
  --large-handoff-chunks 2 \
  --max-handoffs 2 \
  --num-test-questions 20
