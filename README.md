# LEAP

LEAP is an observe-and-rollback scheduler for efficient mathematical reasoning.
The current mainline uses a small model to generate chunk-level reasoning, probes
each chunk for risk, temporarily hands off risky regions to a larger model, and
then returns control to the small model when the trajectory looks stable again.

This README tracks the experimental mainline. Older notes may lag behind the
latest JSON traces; when in doubt, treat `result/**/*.json` as the source of
truth.

## Current Mainline Snapshot

As of 2026-05-22, the paper-facing mainline is:

- Task family: math reasoning scheduling
- Small model: `Qwen2.5-1.5B`
- Large model for GSM8K/SVAMP: `Qwen2.5-7B`
- Large model for current MATH500 pilot: `Qwen2.5-32B`
- Answer format: boxed final answer protocols where available
- Probe feature spec: `boundary+mean`
- Scheduler: adaptive observe-and-rollback
- Default handoff budget: `max_handoffs=2`
- Main comparison axes: final-answer accuracy, raw parameter-weighted token cost, and wall-clock latency

The strongest current result is the GSM8K boxed 1.5B-to-7B adaptive scheduler:

- Trace: `result/traces/observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json`
- Threshold: `0.25`
- Questions: `300`
- Small-only accuracy in scheduler split: `0.7467`
- Scheduled accuracy: `0.8867`
- Gain over small: `+0.1400`
- Trigger rate: `1.0000`
- Avg handoffs/question: `1.950`
- Avg parameter-weighted token cost: `1090.61`
- Mean latency: `10.85s/question`

This is the mainline setting to extend unless an experiment explicitly says it
is an ablation.

## Best Current Results

### LEAP Scheduler

| Dataset | Setting | Trace | Threshold | N | Small Acc | LEAP Acc | Gain | Trigger Rate | Handoffs / Question | Avg Raw Cost | Mean Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 1.5B -> 7B, boxed adaptive | `result/traces/observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.25 | 300 | 0.7467 | 0.8867 | +0.1400 | 1.0000 | 1.950 | 1090.61 | 10.85s |
| SVAMP | 1.5B -> 7B, boxed adaptive SVAMP thresholds | `result/traces/observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.30 | 300 | 0.8100 | 0.8600 | +0.0500 | 0.8933 | 1.323 | 651.63 | 7.77s |
| MATH500 | 1.5B -> 32B, clean adaptive pilot | `result/traces/observe_rollback_traces_math500_vllm_hidden_only_t2048_adaptive_clean055.json` | 0.55 | 100 | 0.7200 | 0.7800 | +0.0600 | 0.7000 | 1.180 | n/a | n/a |

### Best Scheduling Configs

| Dataset | Primary Trace | Best Threshold | Small Model | Large Model | Backend | Answer Type | Scheduler Config | Selection Criterion |
|---|---|---:|---|---|---|---|---|---|
| GSM8K | `result/traces/observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.25 | `Qwen2.5-1.5B` | `Qwen2.5-7B` | HF | `gsm8k_boxed_numeric` | adaptive observe-and-rollback, `max_handoffs=2`, `large_handoff_chunks=2`, `boundary+mean` probe | Best scheduled accuracy in current GSM8K sweep |
| SVAMP | `result/traces/observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.30 | `Qwen2.5-1.5B` | `Qwen2.5-7B` | HF | `svamp_boxed_numeric` | adaptive observe-and-rollback, `max_handoffs=2`, `large_handoff_chunks=2`, `boundary+mean` probe | Best scheduled accuracy in current SVAMP sweep |

### Threshold Sweeps

GSM8K primary adaptive sweep:

| Trace | Threshold | N | Small Acc | Scheduled Acc | Gain | Trigger Rate | Handoffs / Question | Avg Raw Cost | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.15 | 300 | 0.7467 | 0.8600 | +0.1133 | 1.0000 | 1.983 | 1084.82 | 10.20s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.20 | 300 | 0.7467 | 0.8700 | +0.1233 | 1.0000 | 1.980 | 1091.82 | 9.83s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.25 | 300 | 0.7467 | 0.8867 | +0.1400 | 1.0000 | 1.950 | 1090.61 | 10.85s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.30 | 300 | 0.7467 | 0.8800 | +0.1333 | 1.0000 | 1.920 | 1084.07 | 11.29s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json` | 0.35 | 300 | 0.7467 | 0.8633 | +0.1167 | 1.0000 | 1.877 | 1064.32 | 11.14s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.40 | 300 | 0.7467 | 0.8367 | +0.0900 | 1.0000 | 1.803 | 1025.99 | 13.60s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.50 | 300 | 0.7467 | 0.8267 | +0.0800 | 0.9833 | 1.490 | 855.40 | 10.24s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.60 | 300 | 0.7467 | 0.7900 | +0.0433 | 0.6400 | 0.843 | 601.57 | 6.93s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.70 | 300 | 0.7467 | 0.7433 | -0.0033 | 0.0867 | 0.107 | 430.23 | 4.46s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.80 | 300 | 0.7467 | 0.7467 | +0.0000 | 0.0167 | 0.017 | 402.34 | 4.27s |

GSM8K consecutive-risk ablation:

| Trace | Threshold | N | Small Acc | Scheduled Acc | Gain | Trigger Rate | Handoffs / Question | Avg Raw Cost | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_consecutive.json` | 0.15 | 300 | 0.7467 | 0.8667 | +0.1200 | 1.0000 | 1.967 | 1099.83 | 11.35s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_consecutive.json` | 0.20 | 300 | 0.7467 | 0.8567 | +0.1100 | 1.0000 | 1.893 | 1055.90 | 11.00s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_consecutive.json` | 0.25 | 300 | 0.7467 | 0.8367 | +0.0900 | 1.0000 | 1.813 | 1000.49 | 10.29s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_consecutive.json` | 0.30 | 300 | 0.7467 | 0.8433 | +0.0967 | 1.0000 | 1.743 | 959.12 | 11.00s |
| `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_consecutive.json` | 0.35 | 300 | 0.7467 | 0.8233 | +0.0767 | 0.9900 | 1.593 | 894.98 | 12.06s |

SVAMP adaptive sweep:

| Trace | Threshold | N | Small Acc | Scheduled Acc | Gain | Trigger Rate | Handoffs / Question | Avg Raw Cost | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.30 | 300 | 0.8100 | 0.8600 | +0.0500 | 0.8933 | 1.323 | 651.63 | 7.77s |
| `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.35 | 300 | 0.8100 | 0.8533 | +0.0433 | 0.7933 | 1.120 | 599.56 | 8.09s |
| `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.40 | 300 | 0.8100 | 0.8533 | +0.0433 | 0.7033 | 0.940 | 550.41 | 7.11s |
| `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.45 | 300 | 0.8100 | 0.8467 | +0.0367 | 0.5133 | 0.683 | 491.17 | 5.20s |
| `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json` | 0.50 | 300 | 0.8100 | 0.8400 | +0.0300 | 0.3533 | 0.460 | 436.58 | 4.65s |

### Useful Cost/Latency Variants

| Dataset | Setting | Trace | Threshold | N | LEAP Acc | Gain | Trigger Rate | Avg Raw Cost | Mean Latency | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GSM8K | monotonic wide | `result/traces/observe_rollback_traces_gsm8k_15b_to_7b_boxed_monotonic_wide.json` | 0.40 | 300 | 0.8133 | +0.0667 | 0.9967 | 760.00 | 5.76s | Faster/lower-cost ablation, lower accuracy |
| GSM8K | high-threshold adaptive | `result/traces/observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive_highthr.json` | 0.50 | 300 | 0.8267 | +0.0800 | 0.9833 | 855.40 | 10.24s | Accuracy/cost tradeoff |
| SVAMP | all-HF sweep | `result/traces/observe_rollback_traces_svamp_15b_to_7b_boxed_allhf_sweep5.json` | 0.70 | 300 | 0.8467 | +0.0367 | 0.6767 | 415.63 | 2.59s | Best cost-efficient SVAMP point |
| SVAMP | math500-style rule | `result/traces/observe_rollback_traces_svamp_15b_to_7b_boxed_allhf_math500_rule.json` | 0.55 | 300 | 0.8400 | +0.0300 | 0.4433 | 415.00 | 2.80s | Lower trigger rate, same cost band |

### Model-Only References

| Dataset | Model | File | N | Accuracy | Avg Raw Cost | Mean Latency |
|---|---|---|---:|---:|---:|---:|
| GSM8K | Qwen2.5-1.5B | `result/analysis_outputs/qwen25_15b_only_gsm8k_test_boxed.json` | 300 | 0.7067 | 466.78 | 6.05s |
| GSM8K | Qwen2.5-7B | `result/analysis_outputs/qwen25_7b_only_gsm8k_test_boxed.json` | 300 | 0.9400 | 2098.86 | 7.02s |
| SVAMP | Qwen2.5-1.5B | `result/analysis_outputs/qwen25_15b_only_svamp_test_boxed_hf_rerun.json` | 300 | 0.8100 | 314.54 | 2.04s |
| SVAMP | Qwen2.5-7B | `result/analysis_outputs/qwen25_7b_only_svamp_test_boxed_hf.json` | 300 | 0.9267 | 1455.23 | 4.51s |
| MATH500 | Qwen2.5-32B | `result/analysis_outputs/qwen25_32b_only_math500_test100_vllm_clean055.json` | 100 | 0.8000 | n/a | n/a |
| MATH500 | Qwen2.5-Math-7B-Instruct | `result/analysis_outputs/qwen25_math_7b_only_math500_test100_hf_clean055.json` | 100 | 0.8400 | n/a | n/a |

### External Baselines

| Dataset | Baseline | File | N | Accuracy | Avg Raw Cost | Mean Latency |
|---|---|---|---:|---:|---:|---:|
| GSM8K | GlimpRouter, 1.5B/7B all-HF boxed | `result/baselines/glimprouter_gsm8k_qwen15b_qwen7b_allhf_boxed_20260521_210321/glimprouter_benchmark_summary.json` | 300 | 0.6367 | 958.16 | 6.82s |
| SVAMP | GlimpRouter, 1.5B/7B all-HF boxed | `result/baselines/glimprouter_svamp_qwen15b_qwen7b_allhf_20260520_140542/glimprouter_benchmark_summary.json` | 300 | 0.8467 | 825.71 | 3.28s |
| SVAMP | GlimpRouter, 1.5B/7B all-vLLM boxed | `result/baselines/glimprouter_svamp_qwen15b_qwen7b_all_vllm_20260520_121547/glimprouter_benchmark_summary.json` | 300 | 0.8567 | 2135.60 | 8.13s |
| MATH500 | GlimpRouter, 1.5B/32B | `result/baselines/glimprouter_math500_qwen15b_qwen32b_20260519_142014/glimprouter_benchmark_summary.json` | 500 | 0.2820 | 2446.03 | 5.71s |

## Interpretation

The current AAAI-facing story is:

- GSM8K: LEAP substantially improves over the 1.5B model and strongly beats the current GlimpRouter run, while using about half of the 7B-only parameter-weighted token cost.
- SVAMP: LEAP reaches the best observed 1.5B-to-7B GlimpRouter accuracy band, with a stronger cost profile in the cost-efficient settings.
- MATH500: the current result is a 100-question pilot. It is useful as a hard-dataset signal, but should be expanded before becoming the primary paper result.

## Experimental Workflow

The main experiment pipeline is:

1. Build chunk-level trajectories with the small model.
2. Label chunks with a strict large-model judge.
3. Train a probe on chunk features.
4. Run model-only baselines.
5. Run the observe-and-rollback scheduler.
6. Export trace JSON and summarize accuracy, cost, latency, trigger rate, and handoff behavior.

Core entrypoints:

- `core_package/pipelines/build_dataset.py`
- `core_package/pipelines/referee_32b_labeling_strict.py`
- `core_package/probes/train_probe_artifact_torch.py`
- `core_package/schedulers/simulate_observe_rollback_scheduler.py`
- `evaluation/evaluate_model_only_accuracy.py`
- `evaluation/benchmark_latency_compare.py`

Baseline adapters:

- `experimental/baselines/GlimpRouter/src/run_leap_benchmarks.py`
- `experimental/baselines/RSD/run_leap_benchmarks.py`

## Repository Structure

- `core_package/`: main pipeline, probe, scheduler, answer extraction, and shared config
- `dataset/`: raw datasets, local JSONL splits, audits, and large local trajectory/label artifacts when present
- `evaluation/`: accuracy evaluation, cost/latency analysis, plotting, and failure analysis
- `result/`: scheduler traces, model-only JSON outputs, figures, and baseline summaries
- `experimental/baselines/`: external baseline integrations, including GlimpRouter, R2R, and the in-progress RSD reproduction
- `unsorted/`: legacy code and older experiment notes

## Evaluation Metrics

LEAP reports three complementary metrics. Accuracy is the primary quality metric;
compute cost and latency describe two different notions of efficiency.

### 1. Final-Answer Accuracy

Accuracy is question-level final-answer correctness:

- `small-only`: whether the small model's final answer is correct.
- `large-only`: whether the large model's final answer is correct.
- `scheduled`: whether the final answer after LEAP routing is correct.
- `gain over small = scheduled accuracy - small-only accuracy`.

This is the main task metric used in result tables.

### 2. Raw Compute Cost

The result tables above report `avg_param_weighted_token_cost` when available.
This is a raw, unnormalized proxy for inference compute:

```text
raw parameter-weighted token cost ~= generated tokens * model parameter count
```

Lower is better. These values have not been divided by the large-only cost. For
example, a LEAP raw cost of `1090.61` should be compared directly with the
corresponding large-only raw cost, or converted into a ratio manually.

For paper-facing compute plots, also report the normalized FLOPs ratio defined in
`evaluation/LATENCY_AND_COST.md`:

```text
normalized FLOPs ratio = scheduled total inference FLOPs / large-only total inference FLOPs
```

The normalized FLOPs estimate should include small-model decoding, rollback
waste, large-model prefix rebuild/prefill, and large-model decoding. This is the
recommended compute metric for paper figures and tables.

### 3. Wall-Clock Latency

Latency is end-to-end seconds per question under a fixed backend and hardware
stack:

- `latency_mean_s`
- `latency_median_s`
- `latency_p90_s`
- `sec_per_question_mean`

Latency is not interchangeable with FLOPs. Two methods can have similar FLOPs
but different latency because routing can introduce repeated handoffs, repeated
prefill, controller overhead, and backend request overhead.

The 2026-05-22 GSM8K and SVAMP runs are intended to be compared within the same
HF, same-hardware, batch-size-1 setting. MATH500 pilot numbers use a different
serving setup in places, so avoid mixing its latency with the GSM8K/SVAMP latency
claims unless the run is repeated under the same backend and hardware.

## Mainline Policy

Unless an experiment is explicitly marked as an ablation, new work should build
from the 2026-05-22 adaptive observe-and-rollback mainline:

- GSM8K primary setting: `observe_rollback_traces_gsm8k_15b_to_7b_boxed_adaptive.json`, threshold `0.25`
- SVAMP primary setting: `observe_rollback_traces_svamp_15b_to_7b_boxed_adaptive_svamp_thresholds.json`, threshold `0.30`
- MATH500 pilot setting: `observe_rollback_traces_math500_vllm_hidden_only_t2048_adaptive_clean055.json`, threshold `0.55`

When adding a new result, update this README and keep the trace JSON in
`result/traces/` with a descriptive filename.
