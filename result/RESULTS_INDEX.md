# Results Index

This file summarizes the result artifacts currently kept in the repository after cleanup.

## Model-Only Results

| Dataset | Split | Small Model File | Small Acc | Large Model File | Large Acc | Notes |
|---|---|---|---:|---|---:|---|
| GSM8K | heldout100 | `analysis_outputs/qwen25_15b_only_heldout100_newextract.json` | 0.7900 | `analysis_outputs/qwen25_32b_only_heldout100_newextract.json` | 0.9400 | Main GSM8K baseline kept after extractor fix |
| SVAMP | test300 | `analysis_outputs/qwen25_15b_only_svamp_test_protocol.json` | 0.7767 | `analysis_outputs/qwen25_32b_only_svamp_test.json` | 0.8533 | Current SVAMP protocol version |
| MATH500 | test500 | `analysis_outputs/qwen25_15b_only_math500_boxed.json` | 0.4100 | `analysis_outputs/qwen25_32b_only_math500_boxed.json` | 0.6300 | Old boxed protocol with plain Qwen2.5 models; expected to be replaced by Qwen2.5-Math results |

## Scheduler Results

| Dataset | Trace File | Thresholds Kept | Best Threshold | Small Acc | Scheduled Acc | Gain | Notes |
|---|---|---|---:|---:|---:|---:|---|
| GSM8K | `traces/observe_rollback_traces_mainline_rerun_dense.json` | 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60 | 0.25 / 0.35 / 0.40 | 0.7900 | 0.8800 | +0.0900 | Main GSM8K scheduler trace; commonly referenced 0.25 setting is preserved |
| SVAMP | `traces/observe_rollback_traces_svamp_test_mainline_fixed.json` | 0.25, 0.40, 0.50 | 0.50 | 0.8200 | 0.8633 | +0.0433 | Current fixed SVAMP scheduler trace |
| MATH500 | `traces/observe_rollback_traces_math500_vllm_hidden_only_t2048_adaptive_clean055.json` | 0.55 | 0.55 | 0.7200 | 0.7800 | +0.0600 | Current clean adaptive baseline with `boundary+mean`, adaptive handoff, and consecutive-risk trigger |

## Average Generated Tokens

| Dataset | Small Avg Tokens | Large Avg Tokens |
|---|---:|---:|
| GSM8K heldout100 | 299.85 | 306.66 |
| SVAMP test300 | 196.56 | 194.12 |
| MATH500 test500 | 501.82 | 522.26 |

## Important Reminder

- GSM8K currently uses the `legacy_math` protocol.
- SVAMP currently uses the `svamp_numeric` protocol.
- MATH500 mainline scheduler now uses the `math500_qwen_boxed` protocol.
- The current kept MATH500 scheduler baseline is the clean adaptive `0.55` setting with `boundary+mean`.
