# Accuracy, Compute Cost, and Latency

This note defines the three reporting axes used in the current LEAP mainline:

- final answer accuracy
- normalized compute cost
- end-to-end latency

It is intended to keep internal experiments and external baseline comparisons on a consistent footing.

## 1. Accuracy

The primary metric is **final answer accuracy** at the question level.

- For `small-only`, we evaluate whether the SLM final answer is correct.
- For `scheduled`, we evaluate whether the final answer after observe-and-rollback routing is correct.
- For `large-only`, we evaluate whether the LLM final answer is correct.

For scheduler traces, the summary metrics are:

- `small_only_accuracy`
- `scheduled_accuracy`
- `scheduled_gain_over_small = scheduled_accuracy - small_only_accuracy`

These are question-level end-to-end metrics, not chunk-level probe metrics.

## 2. Compute Cost

The current normalized cost analysis is implemented in:

- [plot_threshold_accuracy_flops_compare.py](./plot_threshold_accuracy_flops_compare.py)

This script supports three modes:

- `token_proxy`
- `approx_flops`
- `rsd_flops`

The recommended mainline mode is:

- `approx_flops`

### 2.1 What `approx_flops` includes

For each question, total scheduled cost is approximated as the sum of:

- SLM decode cost
- rollback waste cost
- LLM prefix rebuild (prefill) cost
- LLM decode cost
- optional probe cost

The script then normalizes this by the cost of an `LLM-only` run on the same question:

- `overall_flops_ratio = total_scheduled_cost / llm_only_cost`

### 2.2 Recommended wording

When reporting compute, use:

> Normalized total inference FLOPs relative to the LLM-only baseline, including SLM decode, rollback waste, LLM prefill, and LLM decode.

This is the cleanest current repository-wide compute definition.

## 3. Latency

Latency is **not** the same as FLOPs.

The recommended latency metric is:

> End-to-end per-question wall-clock latency, measured from the start of prompt processing to final answer completion, under batch size 1 on the same hardware/software stack.

### 3.1 Why FLOPs are not enough

Different routing methods can have similar FLOPs but very different wall-clock behavior because of:

- repeated handoffs
- repeated prefill
- token-by-token or step-by-step control overhead
- backend request overhead

So FLOPs should be treated as compute cost, not as latency.

### 3.2 How to measure latency in this repo

Use:

- [benchmark_latency_compare.py](./benchmark_latency_compare.py)

This script measures real wall-clock time by re-running each method command.

It reports:

- `latency_mean_s`
- `latency_median_s`
- `latency_p90_s`
- `sec_per_question_mean`
- `qps_mean`
- `latency_ratio_vs_baseline`
- `accuracy_delta_vs_baseline`
- `gain_per_extra_second`

### 3.3 Important limitation

If `--skip-run` is used, the script will summarize result files but **cannot recover real latency** from old traces.

That means:

- use normal mode for true latency
- use `--skip-run` only for result aggregation

## 4. Recommended Reporting Template

For internal and paper-facing comparisons, the cleanest table layout is:

### Main results

- Method
- Accuracy
- FLOPs ratio
- Avg latency
- P90 latency

### Behavior / routing analysis

- Trigger rate
- Avg handoff count
- Avg large takeover tokens
- Avg trigger progress

## 5. Example benchmark config

An example config for real latency benchmarking is provided at:

- [latency_benchmark.example.json](./latency_benchmark.example.json)

Typical usage:

```bash
python -m evaluation.benchmark_latency_compare \
  --config-path evaluation/latency_benchmark.example.json \
  --output-path result/analysis_outputs/latency_math500_summary.csv \
  --repeats 3
```

If you only want to read existing results:

```bash
python -m evaluation.benchmark_latency_compare \
  --config-path evaluation/latency_benchmark.example.json \
  --output-path result/analysis_outputs/latency_math500_summary.csv \
  --skip-run
```

In `--skip-run` mode, latency columns will be unavailable unless the methods are rerun.
