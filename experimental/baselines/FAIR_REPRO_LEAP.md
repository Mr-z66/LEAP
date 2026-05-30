# LEAP Baseline Reproduction

This directory keeps external baseline code intact and adds thin LEAP adapters for comparable runs.

## Recommended order

1. Run GlimpRouter first. It already has a LEAP adapter and only needs the small/large model paths.
2. Run RSD after the PRM is available locally or served by vLLM.
3. Treat R2R separately. The imported R2R tree does not include `resource/default_router.pt`, and its default evaluation config does not cover GSM8K/SVAMP/MATH500.

## One-command sweep

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate care_env
cd /root/LEAP

DATASETS="gsm8k,svamp" \
MAX_QUESTIONS=300 \
SMALL_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B \
LARGE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B \
PRM_MODEL_PATH=/root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
GLIMP_THRESHOLDS="0.6 0.8 1.0 1.2 1.4" \
RSD_THRESHOLDS="0.6 0.7 0.8" \
bash experimental/baselines/run_fair_repro.sh
```

## Outputs

- `result/baselines/glimprouter_qwen15b_qwen7b_thr*/glimprouter_benchmark_summary.csv`
- `result/baselines/rsd_qwen15b_qwen7b_prm*/rsd_benchmark_summary.csv`
- `result/baselines/fair_repro_summary.csv`
- `result/baselines/fair_repro_summary.json`

## Fairness notes

- Use the same `Qwen2.5-1.5B -> Qwen2.5-7B` model pair as the mixed scheduler.
- Use LEAP answer extractors through the adapters.
- Report raw parameter-weighted token cost, not normalized cost, then normalize only in plotting.
- GlimpRouter always uses the large model for final answer generation in the imported implementation.
- RSD includes PRM score calls in cost as `prm_params_b * score_calls`.
- R2R requires either downloading a matching router checkpoint or training a router for the exact model pair before it is a fair GSM8K/SVAMP/MATH500 comparison.

