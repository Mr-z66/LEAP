# Preliminary Plan 2026-05-28

This is the current preliminary experiment plan for the mixed-domain LEAP
mainline after the latest GSM8K/SVAMP/MATH500 scheduler runs.

Important structure decision:

The paper should use `Preliminary` as motivation, not as the main method result.
The preliminary section should show why surface-level uncertainty functions are
insufficient for detecting wrong reasoning chunks, and why hidden-state probes
are worth introducing. Scheduler accuracy, fixed4 handoff results, and
rescue/harm analysis belong in the main `Experiments` section after `Method`.

Preliminary should make one focused claim:

> Surface-level token uncertainty signals such as entropy, top-1 probability,
> and margin do not reliably separate correct and erroneous reasoning chunks.
> Hidden states provide a stronger signal, motivating a learned probe for
> observe-and-rollback routing.

## Current State

Latest summary files:

- `result/analysis_outputs/new_four_trace_summary.csv`
- `result/analysis_outputs/new_four_trace_details.json`
- `result/analysis_outputs/current_test_scheduler_summary.csv`
- `result/analysis_outputs/current_test_scheduler_details.json`

Latest important traces:

- `result/traces/observe_rollback_traces_mixed_fixed4_sweep015_050_gsm8k_test.json`
- `result/traces/observe_rollback_traces_mixed_fixed4_sweep015_050_svamp_test.json`
- `result/traces/observe_rollback_traces_mixed_original_takeover4_thr015_045_gsm8k_test.json`
- `result/traces/observe_rollback_traces_mixed_original_takeover4_thr015_045_svamp_test.json`
- `result/traces/observe_rollback_traces_mixed_rewrite_current_chunk_thr040_unlimited_gsm8k_test.json`
- `result/traces/observe_rollback_traces_mixed_rewrite_current_chunk_thr040_unlimited_svamp_test.json`
- `result/traces/observe_rollback_traces_mixed_adaptive4_thr040_math500_math500_test.json`

Important local caveat:

- `result/artifacts/` is not present in this workspace snapshot, so artifact-level
  probe reruns must be performed on the training server or after syncing
  artifacts.

## Preliminary Scope

The preliminary experiment should answer this question:

> Before designing LEAP, can we detect when a small model's reasoning has gone
> wrong using simple signals available during generation?

The answer should be:

1. Raw token uncertainty is too weak or unstable.
2. Hidden states carry richer information about reasoning validity.
3. Therefore, LEAP should use a hidden-state probe rather than a hand-written
   entropy threshold.

Do not use preliminary to argue that the final scheduler is best. That comes
later.

## Later Experiment Context

The latest scheduler results are still important, but they should be treated as
post-method experimental results.

### Current Main Candidate

Use `mixed_fixed4_sweep015_050` as the current strongest preliminary scheduler
setting.

Best points from `new_four_trace_summary.csv`:

| Dataset | Trace | Threshold | Small Acc | Scheduled Acc | Gain | Trigger Rate | Fixed Wrong | Harmed Correct | Avg Cost | Mean Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | `mixed_fixed4_sweep015_050` | 0.15 | 0.6900 | 0.8500 | +0.1600 | 0.9967 | 55 | 7 | 1497.90 | 7.22s |
| SVAMP | `mixed_fixed4_sweep015_050` | 0.20 | 0.8033 | 0.8933 | +0.0900 | 0.9600 | 34 | 7 | 865.91 | 4.87s |

Interpretation:

- GSM8K: the scheduler fixes 55 of 93 small-model wrong questions and harms 7
  originally correct questions at the best-accuracy point.
- SVAMP: the scheduler fixes 34 of 59 small-model wrong questions and harms 7
  originally correct questions at the best-accuracy point.
- These two rows are currently the cleanest preliminary result for the mixed
  GSM8K+SVAMP story.

### Controller Ablations

These are useful for positioning, but should not replace the main candidate.

| Setting | Dataset | Threshold | Scheduled Acc | Gain | Notes |
|---|---|---:|---:|---:|---|
| original takeover4 | GSM8K | 0.15 | 0.8300 | +0.1400 | Slightly below fixed4. |
| original takeover4 | SVAMP | 0.15 | 0.8900 | +0.0867 | Similar to fixed4, slightly lower than best fixed4. |
| rewrite current chunk | GSM8K | 0.40 | 0.7633 | +0.0733 | Too weak for main result. |
| rewrite current chunk | SVAMP | 0.40 | 0.8800 | +0.0767 | Competitive but still below fixed4 best. |
| adaptive4 MATH500 | MATH500 | 0.40 | 0.5567 | +0.0333 | Exploratory only. |
| boundary t1536 fixed2 MATH500 | MATH500 | 0.50 | 0.5167 | -0.0067 | Not useful as a positive result. |

Preliminary takeaway:

- Fixed-length large handoff with four chunks is currently stronger than the
  single current-chunk rewrite ablation.
- MATH500 is still exploratory and should not be central unless the extractor,
  labels, or handoff policy improves.

## Paper Story

Recommended full-paper story:

1. Start with chunk-level evidence: hidden states encode reasoning validity.
2. Show that raw uncertainty alone is weaker or less stable.
3. Train one mixed-domain probe on GSM8K+SVAMP calibration data.
4. Run the same scheduler policy on held-out GSM8K and SVAMP.
5. Show accuracy gains and rescue/harm counts.
6. Discuss MATH500 as an early hard-domain pilot, not a headline result.

One-sentence positioning:

> LEAP uses hidden-state probes to detect local reasoning failures and performs
> observe-and-rollback handoffs to a larger model, improving small-model
> reasoning accuracy under an explicit handoff budget.

## Preliminary Experiments To Prioritize

### Experiment P1: Surface Signals vs Hidden States

Goal:

- Show that raw uncertainty signals are weak for chunk-level error detection.
- Show that hidden states separate correct and erroneous chunks better.
- Motivate why the method uses a learned hidden-state probe.

Run on:

- GSM8K calibration labels.
- SVAMP calibration labels.
- Mixed GSM8K+SVAMP calibration labels.

Main command template:

```bash
export OMP_NUM_THREADS=1

python -m evaluation.compare_latent_uncertainty_signals \
  --label-path <LABEL_PATH> \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix <OUTPUT_PREFIX>
```

Concrete commands:

```bash
python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix gsm8k_calib_signal_comparison_20260528
```

```bash
python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/svamp_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix svamp_calib_signal_comparison_20260528
```

```bash
python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_gsm8k_svamp_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix mixed_gsm8k_svamp_signal_comparison_20260528
```

Report:

- Error AUROC.
- Error AUPRC.
- Best error F1.
- Recall@5%, Recall@10%, Recall@20%.
- Precision@5%, Precision@10%, Precision@20%.

Decision rule:

- If `boundary+mean` wins or is consistently top-tier, use it as the default
  preliminary signal.
- If `boundary` and `mean` split by dataset, report `boundary+mean` as the robust
  combined signal.

### Experiment P2: Visual Separability

Goal:

- Produce one figure that makes the preliminary motivation obvious.
- Show score distributions for correct chunks vs error chunks.

Recommended figure:

- Left: entropy or margin score distributions for correct/error chunks.
- Right: `boundary+mean` probe score distributions for correct/error chunks.

Preferred dataset:

- GSM8K calibration split first, because the labels and prior artifacts are
  already most stable.

Interpretation to aim for:

> Entropy assigns overlapping scores to correct and erroneous reasoning chunks,
> while the hidden-state probe shifts erroneous chunks toward the high-risk
> region. This motivates hidden-state probing as the routing signal.

### Experiment P3: Trigger-Budget Motivation

Goal:

- Match the scheduler motivation without presenting scheduler results yet.
- Ask: under a limited handoff budget, which signal catches more true error
  chunks?

Use:

- Recall@trigger 5%, 10%, 20%.
- Precision@trigger 5%, 10%, 20%.

Paper role:

- This bridges preliminary to Method: if hidden probes rank error chunks better,
  a scheduler can use the ranking to decide when to hand off.

## Post-Method Experiments To Keep

These are not preliminary. Keep them for the main experiment section.

### Experiment E1: Mixed Probe End-To-End Scheduling

Goal:

- Use the latest fixed4 traces to show actual final-answer improvement.

Main rows:

- GSM8K fixed4 threshold 0.15.
- SVAMP fixed4 threshold 0.20.

Report table:

| Dataset | Method | Small Acc | Scheduled Acc | Gain | Trigger Rate | Avg Handoffs | Avg Cost | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | mixed fixed4 | 0.6900 | 0.8500 | +0.1600 | 0.9967 | 1.4047 | 1497.90 | 7.22s |
| SVAMP | mixed fixed4 | 0.8033 | 0.8933 | +0.0900 | 0.9600 | 1.1597 | 865.91 | 4.87s |

Optional comparison rows:

- Old `mixed_probe_gsm8k_test`: 0.8100 at threshold 0.25.
- Old `mixed_probe_svamp_test`: 0.8833 at threshold 0.40.
- Original takeover4 best.
- Rewrite-current-chunk ablation.

Decision rule:

- Keep fixed4 as main if the paper needs a single clear setting.
- Use original takeover4 and rewrite-current-chunk as ablations, not the primary
  result.

### Experiment E2: Rescue And Harm Analysis

Goal:

- Explain what the scheduler is doing, not just report accuracy.

Use fields from `new_four_trace_summary.csv` and examples from
`new_four_trace_details.json`.

For the main rows:

| Dataset | Threshold | Small Wrong | Triggered Wrong | Fixed Wrong | Harmed Correct | First Trigger Median Progress |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.15 | 93 | 93 | 55 | 7 | 0.1628 |
| SVAMP | 0.20 | 59 | 58 | 34 | 7 | 0.2627 |

Figure ideas:

- Stacked bars: fixed wrong, stubborn wrong, harmed correct.
- Trigger-rate vs scheduled accuracy.
- First-trigger progress histogram.

Paper wording:

> Most accuracy gains come from correcting small-model failures rather than
> merely changing formatting. The controller triggers early in the reasoning
> trajectory on many failures, giving the large model enough context to redirect
> the solution.

### Experiment E3: Cost/Accuracy Frontier

Goal:

- Show that threshold controls an explicit accuracy-cost tradeoff.

Use:

- `observe_rollback_traces_mixed_fixed4_sweep015_050_gsm8k_test.json`
- `observe_rollback_traces_mixed_fixed4_sweep015_050_svamp_test.json`

Plot:

- X-axis: average parameter-weighted token cost.
- Y-axis: scheduled accuracy.
- Annotate thresholds.

Expected pattern:

- Low threshold: highest accuracy, high trigger rate and cost.
- Higher threshold: lower cost, lower gain.

This is important because the current best-accuracy GSM8K point has a very high
trigger rate. The paper should present it as an accuracy-cost tradeoff, not as a
free win.

### Experiment E4: MATH500 Pilot

Goal:

- Decide whether MATH500 belongs in the main preliminary section or only in an
  appendix/pilot paragraph.

Current evidence:

- `mixed_adaptive4_thr040_math500`: 0.5233 small to 0.5567 scheduled, gain +0.0333.
- `math500_boundary_t1536_single_h1_fixed2`: negative gain.

Decision:

- Keep MATH500 as exploratory unless a new run reaches a clearer gain.
- Do not let MATH500 dilute the stronger GSM8K/SVAMP mixed-domain story.

## Low-Entropy Weighting Decision

Current decision:

- Do not use low-entropy hard-error weighting as the mixed mainline default.

Observed pattern:

- GSM8K-only probe benefited from low-entropy weighting.
- Mixed GSM8K+SVAMP probe got worse on mixed calibration, GSM8K calibration, and
  SVAMP calibration under `boundary+mean`.

Interpretation:

- Low-entropy hard errors are not stable enough across the current mixed domains.
- Keep this as an ablation note, not a core method.

## What To Write In Preliminary

Suggested preliminary subsection structure:

1. `Surface Uncertainty Is Insufficient`
   - Show entropy/top1/margin results.
   - Main claim: these scalar signals do not reliably separate wrong reasoning
     chunks from correct chunks.

2. `Hidden States Encode Reasoning Validity`
   - Show boundary, mean, and boundary+mean results.
   - Main claim: hidden states give a much stronger chunk-error signal.

3. `Motivation For LEAP`
   - Connect separability to routing.
   - Main claim: a learned hidden-state probe is a better trigger than a
     hand-written entropy threshold.

Avoid in preliminary:

- Scheduler accuracy tables.
- Fixed4 vs takeover4 comparisons.
- Rescue/harm counts.
- MATH500 pilot claims.

## Tonight's Updated Checklist

- [ ] Run or refresh signal comparison tables for GSM8K, SVAMP, and mixed labels.
- [ ] Build one preliminary table: raw uncertainty vs hidden-state probes.
- [ ] Build one visual separability figure: entropy vs `boundary+mean`.
- [ ] Build one trigger-budget table or curve: Recall@trigger-rate.
- [ ] Write 1-2 preliminary paragraphs that motivate hidden-state probing.
- [ ] Move scheduler/fixed4/rescue-harm material to the later experiment plan.

## Open Questions

- Are the current fixed4 traces using the exact probe artifact intended for the
  paper-facing mixed mainline?
- Do we have synchronized test labels for GSM8K/SVAMP, or should signal
  comparison remain calibration split only for now?
- Should the main table optimize for accuracy, cost, or a fixed trigger-rate
  operating point?
- Do we want the headline setting to be one fixed threshold across GSM8K and
  SVAMP, or one threshold selected per dataset?
- For preliminary specifically, should we show one dataset in the figure and
  all datasets in the table, or show all three datasets visually?
