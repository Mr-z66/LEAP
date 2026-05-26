# Preliminary Signal Experiments

This note tracks the preliminary experiments for the mixed-domain LEAP mainline.
The goal is to establish that hidden states around reasoning chunks carry a
strong signal for detecting incorrect reasoning prefixes, and that this signal
is stronger than common token-uncertainty baselines.

## Core Claim

Boundary and mean hidden states separate correct and erroneous reasoning chunks.
This makes them useful as routing signals for observe-and-rollback scheduling,
where the controller needs to detect risky chunks before the final answer.

Working paper wording:

> Hidden states at chunk boundaries encode whether the current reasoning prefix
> is still valid. A lightweight probe over boundary and mean hidden states
> detects erroneous chunks more reliably than raw token uncertainty signals such
> as entropy, top-1 probability, and margin.

## Experimental Setup

Task:

- Chunk-level error detection.
- Label convention: `label=1` means the prefix/chunk is still correct, and
  `label=0` means an error has appeared.
- Evaluation target: the error class, i.e. `error_label = 1 - label`.

Split policy:

- Use grouped question splits whenever possible, so chunks from the same question
  do not appear in both train and test.
- Until held-out test labels are available, use calibration labels with grouped
  train/test splits for preliminary evidence.
- Full-calibration evaluation is allowed only as a sanity check and should be
  clearly marked as such.

Main datasets:

- `dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt`
- `dataset/mixed_probe_labels_fallback_second_pass/svamp_calib_labels.pt`
- `dataset/mixed_gsm8k_svamp_calib_labels.pt`

## Signals To Compare

Raw uncertainty baselines:

- `entropy`: final-token entropy.
- `neg_top1_prob`: negative final-token top-1 probability.
- `neg_margin`: negative final-token top-1/top-2 margin.

Learned hidden-state probes:

- `boundary`: boundary hidden state.
- `mean`: mean hidden state over the chunk.
- `boundary+mean`: concatenated boundary and mean hidden states.

Optional follow-up signals:

- `boundary+mean+entropy+top1_prob+margin`
- `relative_position+boundary+mean`
- `delta_prev+boundary+mean`

## Metrics

Report these as the main preliminary metrics:

- Error AUROC.
- Error AUPRC.
- Best error F1.
- Error precision and recall at best F1.
- Recall@trigger-rate for trigger rates `5%`, `10%`, and `20%`.
- Precision@trigger-rate for the same trigger rates.

Why trigger-rate metrics matter:

LEAP does not need a perfect classifier over all chunks. It needs a good ranking
of risky chunks under a limited handoff budget. Recall@trigger-rate directly
answers how many true error chunks are found when the scheduler is allowed to
trigger on only the top-scoring chunks.

## Main Comparison Commands

### GSM8K

```bash
export OMP_NUM_THREADS=1

python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix gsm8k_calib_signal_comparison
```

### SVAMP

```bash
export OMP_NUM_THREADS=1

python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/svamp_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix svamp_calib_signal_comparison
```

### Mixed GSM8K + SVAMP

```bash
export OMP_NUM_THREADS=1

python -m evaluation.compare_latent_uncertainty_signals \
  --label-path dataset/mixed_gsm8k_svamp_calib_labels.pt \
  --feature-specs boundary,mean,boundary+mean \
  --raw-signals entropy,neg_top1_prob,neg_margin \
  --classifier logreg \
  --trigger-rates 0.05,0.10,0.20 \
  --output-prefix mixed_gsm8k_svamp_signal_comparison
```

Expected outputs:

- `result/analysis_outputs/latent_signal_comparison/*_latent_vs_uncertainty.csv`
- `result/analysis_outputs/latent_signal_comparison/*_latent_vs_uncertainty.json`
- `result/analysis_outputs/latent_signal_comparison/*_latent_vs_uncertainty.md`

## Visualization Plan

Use one dataset first, preferably GSM8K, to make the separation visually obvious.

Figure 1: error-score distribution

- Compare `boundary+mean` hidden probe against an entropy-only probe.
- Plot correct chunks and error chunks as separate distributions.
- Desired visual: hidden probe places error chunks noticeably farther to the high
  risk side than entropy-only.

Figure 2: ROC or PR curve

- Compare `boundary+mean`, entropy, negative top-1 probability, and negative
  margin.
- PR curve is more informative when error chunks are a minority.

Existing helper scripts:

- `evaluation/export_probe_chunk_scores.py`
- `evaluation/plot_gsm8k_probe_ablation.py`

### Train Entropy-Only Probe For Figure

```bash
export OMP_NUM_THREADS=1

python -m core_package.probes.train_probe_artifact_torch \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt \
  --output-path result/artifacts/probe_artifact_gsm8k_entropy_only.pt \
  --feature-key entropy \
  --hidden-layers 128,32 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3
```

### Export Scores

```bash
export OMP_NUM_THREADS=1

python -m evaluation.export_probe_chunk_scores \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt \
  --probe-artifact-path result/artifacts/probe_artifact_gsm8k_only_no_lowent.pt \
  --output-path result/analysis_outputs/gsm8k_hidden_only_chunk_scores.csv \
  --feature-set-name hidden_only \
  --all-questions
```

```bash
export OMP_NUM_THREADS=1

python -m evaluation.export_probe_chunk_scores \
  --label-path dataset/mixed_probe_labels_fallback_second_pass/gsm8k_calib_labels.pt \
  --probe-artifact-path result/artifacts/probe_artifact_gsm8k_entropy_only.pt \
  --output-path result/analysis_outputs/gsm8k_entropy_only_chunk_scores.csv \
  --feature-set-name entropy_only \
  --all-questions
```

### Plot GSM8K Ablation

```bash
python -m evaluation.plot_gsm8k_probe_ablation
```

Expected outputs:

- `result/analysis_outputs/gsm8k_error_score_distribution.png`
- `result/analysis_outputs/gsm8k_error_score_boxplot.png`
- `result/analysis_outputs/gsm8k_error_detection_roc.png`

## Tonight's Checklist

- [ ] Run GSM8K signal comparison with logistic regression.
- [ ] Run SVAMP signal comparison with logistic regression.
- [ ] Run mixed GSM8K+SVAMP signal comparison with logistic regression.
- [ ] Fill the result table below.
- [ ] Decide whether `boundary+mean` is the main preliminary signal.
- [ ] Train entropy-only GSM8K probe for visualization.
- [ ] Export hidden-only and entropy-only chunk scores.
- [ ] Generate preliminary GSM8K separation figures.
- [ ] Record one or two qualitative takeaways for the paper draft.

## Result Table Template

### GSM8K Calibration Split

| Signal | Model | Error AUROC | Error AUPRC | Best Error F1 | Recall@5% | Recall@10% | Recall@20% |
|---|---|---:|---:|---:|---:|---:|---:|
| entropy | raw |  |  |  |  |  |  |
| neg_top1_prob | raw |  |  |  |  |  |  |
| neg_margin | raw |  |  |  |  |  |  |
| boundary | logreg |  |  |  |  |  |  |
| mean | logreg |  |  |  |  |  |  |
| boundary+mean | logreg |  |  |  |  |  |  |

### SVAMP Calibration Split

| Signal | Model | Error AUROC | Error AUPRC | Best Error F1 | Recall@5% | Recall@10% | Recall@20% |
|---|---|---:|---:|---:|---:|---:|---:|
| entropy | raw |  |  |  |  |  |  |
| neg_top1_prob | raw |  |  |  |  |  |  |
| neg_margin | raw |  |  |  |  |  |  |
| boundary | logreg |  |  |  |  |  |  |
| mean | logreg |  |  |  |  |  |  |
| boundary+mean | logreg |  |  |  |  |  |  |

### Mixed GSM8K + SVAMP Calibration Split

| Signal | Model | Error AUROC | Error AUPRC | Best Error F1 | Recall@5% | Recall@10% | Recall@20% |
|---|---|---:|---:|---:|---:|---:|---:|
| entropy | raw |  |  |  |  |  |  |
| neg_top1_prob | raw |  |  |  |  |  |  |
| neg_margin | raw |  |  |  |  |  |  |
| boundary | logreg |  |  |  |  |  |  |
| mean | logreg |  |  |  |  |  |  |
| boundary+mean | logreg |  |  |  |  |  |  |

## Preliminary Interpretation Template

Use this only after the table is filled:

> Across GSM8K, SVAMP, and their mixed calibration split, learned probes over
> final-layer hidden states outperform raw uncertainty signals for chunk-level
> error detection. The strongest single signal is `boundary+mean`, suggesting
> that both the chunk endpoint representation and the chunk-average trajectory
> representation encode complementary information about reasoning validity.
> This supports LEAP's design choice to use hidden-state probes rather than raw
> token confidence for online handoff decisions.

## Notes From Current Low-Entropy Check

Low-entropy hard-error weighting should not be part of the current mixed default.

Observed so far:

- GSM8K-only probe: low-entropy weighting improved calibration-set error
  detection.
- Mixed GSM8K+SVAMP probe: low-entropy weighting hurt the mixed calibration
  split, GSM8K calibration subset, and SVAMP calibration subset under
  `boundary+mean`.

Current decision:

- Keep preliminary signal experiments on unweighted training.
- Treat low-entropy weighting as an ablation note, not a mainline setting.
