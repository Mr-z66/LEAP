# Answer-Aware Stop Experiment

This experiment keeps the original mixed fixed4 scheduler unchanged by default.
When enabled, a question stops immediately after a newly generated large-model
chunk contains a complete, non-empty `\boxed{...}` answer.

## Run a smoke test

```bash
DATASETS="gsm8k_test svamp_test" \
THRESHOLDS="0.15,0.20" \
bash experimental/mixed_probe_mainline/scripts/run_scheduler_eval_15b_to_32b_answer_aware.sh
```

The experiment writes separate traces using the default tag:

```text
mixed_5to1_fixed4_15b_to_32b_hf_answer_aware
```

## Run the original scheduler

Run the original script directly. It does not enable `--answer-aware-stop`.

```bash
bash experimental/mixed_probe_mainline/scripts/run_scheduler_eval_15b_to_32b_hf.sh
```

To use the experiment wrapper while explicitly disabling the new behavior:

```bash
ANSWER_AWARE_STOP=0 \
TRACE_TAG=mixed_5to1_fixed4_15b_to_32b_hf_original_check \
bash experimental/mixed_probe_mainline/scripts/run_scheduler_eval_15b_to_32b_answer_aware.sh
```

## Compare

Compare accuracy, token cost, latency, harmed-correct questions, and the new
`scheduler_stop_answer` trace event. The original trace files are not
overwritten unless `TRACE_TAG` is manually set to an existing tag.
