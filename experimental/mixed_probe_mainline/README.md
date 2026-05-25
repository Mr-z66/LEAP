# Mixed-Domain LEAP Mainline

This folder tracks the main experimental line for training one unified LEAP probe
and one fixed routing policy across math and code datasets.

## Goal

Train a single mixed-domain probe on calibration data from:

- GSM8K
- SVAMP
- MATH500
- LiveCodeBench v5

Then evaluate the same scheduler policy on held-out splits from all datasets.

## Baseline References

- R2R: data construction idea, especially small/large divergence supervision.
- RSD: online step-level speculative routing, especially `step_word="\n\n"`.

## Planned Stages

1. Prepare fixed calibration/test splits.
2. Add answer/checking protocol for `livecodebench_codegen`.
3. Generate small-model trajectories and hidden states.
4. Build step chunks using an RSD-style step boundary.
5. Label chunks with the existing two-stage judge flow, optionally enriched by
   R2R-style continuation divergence.
   Recommended check: run a label-consistency audit on each labeled split before
   merging, and inspect suspicious buckets such as final-wrong/no-error,
   final-correct/has-error, and zero-to-one flips.
6. Merge labeled calibration datasets.
7. Train one mixed-domain probe.
8. Evaluate probe F1/recall on held-out data.
9. Run one fixed scheduler policy across datasets.
10. Compare against small-only, large-only, RSD, and R2R where runnable.

## Directory Layout

```text
experimental/mixed_probe_mainline/
  README.md
  configs/        # fixed split sizes, model paths, thresholds
  scripts/        # reproducible pipeline entrypoints
  audits/         # split/leakage/protocol audit notes
```

Generated datasets, model outputs, probes, and traces should stay in the
repository-level `dataset/` and `result/` folders with clear `mixed_` names.
