# RSD LEAP Benchmark Adapter

This directory contains the upstream Reward-Guided Speculative Decoding (RSD)
code plus a LEAP-specific benchmark adapter:

- `run_leap_benchmarks.py`

The adapter keeps RSD's decoding mechanism intact and standardizes the parts
needed for direct comparison with LEAP and GlimpRouter:

- LEAP dataset loading
- LEAP answer extraction and correctness checks
- boxed GSM8K/SVAMP prompts
- raw parameter-weighted token cost
- latency summary
- JSON/CSV outputs under `result/baselines/rsd`

The upstream bundled evaluation datasets under
`external/qwen25_math_evaluation/data/` are intentionally not vendored here.
For LEAP-comparable experiments, use the repository-level `dataset/` files
through `run_leap_benchmarks.py`.

## Backends

The adapter supports two backends:

- `--backend vllm`: use three OpenAI-compatible vLLM services.
- `--backend hf`: load draft, target, and PRM locally with HuggingFace. This avoids
  serving the Skywork PRM through vLLM.

## Required Services For vLLM

RSD online mode expects three OpenAI-compatible vLLM services:

- draft model, usually `Qwen2.5-1.5B`
- target model, usually `Qwen2.5-7B`
- PRM, usually `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`

The upstream helper scripts are:

```bash
bash scripts/serve_draft_model.sh
bash scripts/serve_target_model.sh
bash scripts/serve_prm.sh
```

Default adapter URLs:

```text
draft:  http://localhost:12340/v1
target: http://localhost:12341/v1
prm:    http://localhost:12342/v1
```

## Smoke Test

From the repository root:

```bash
python experimental/baselines/RSD/run_leap_benchmarks.py \
  --backend vllm \
  --datasets gsm8k,svamp \
  --max-questions 5 \
  --draft-model-path models/Qwen2.5-1.5B \
  --target-model-path models/Qwen2.5-7B \
  --prm-model-path Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
  --prm-threshold 0.7 \
  --max-tokens-per-call 2048
```

If the vLLM served model names differ from the tokenizer/model paths, pass:

```bash
--draft-served-model-name <served-draft-name>
--target-served-model-name <served-target-name>
--prm-served-model-name <served-prm-name>
```

HF smoke test, no vLLM services required:

```bash
python experimental/baselines/RSD/run_leap_benchmarks.py \
  --backend hf \
  --datasets gsm8k \
  --max-questions 5 \
  --draft-model-path /root/autodl-tmp/models/Qwen2.5-1.5B \
  --target-model-path /root/autodl-tmp/models/Qwen2.5-7B \
  --prm-model-path /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
  --prm-threshold 0.7 \
  --max-tokens-per-call 2048
```

## Full LEAP-Comparable Run

```bash
python experimental/baselines/RSD/run_leap_benchmarks.py \
  --backend hf \
  --datasets gsm8k,svamp \
  --draft-model-path /root/autodl-tmp/models/Qwen2.5-1.5B \
  --target-model-path /root/autodl-tmp/models/Qwen2.5-7B \
  --prm-model-path /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
  --prm-threshold 0.7 \
  --max-tokens-per-call 2048 \
  --output-root result/baselines/rsd_gsm8k_svamp_qwen15b_qwen7b_prm07
```

Outputs:

- `result/baselines/.../rsd_benchmark_summary.json`
- `result/baselines/.../rsd_benchmark_summary.csv`
- per-dataset `*_summary.json` with per-question rows

These summary fields are intentionally close to GlimpRouter:

- `accuracy`
- `latency_mean_s`
- `latency_median_s`
- `latency_p90_s`
- `avg_draft_tokens`
- `avg_target_tokens`
- `avg_discarded_draft_tokens`
- `avg_prm_score_calls`
- `avg_target_step_fraction`
- `avg_param_weighted_token_cost`

## Notes

The reported `avg_param_weighted_token_cost` is a raw unnormalized cost proxy:

```text
draft_params * (accepted_draft_tokens + discarded_draft_tokens)
+ target_params * target_tokens
+ prm_params * prm_score_calls
```

It is not divided by large-only cost. Convert to a normalized ratio separately
when producing paper plots.
