# Chunk Decode Router

This workspace isolates the new routing direction:

- decision point: current chunk boundary
- input signal: prefix/prefill hidden state before decoding the current chunk
- action: choose `SLM decode` or `LLM decode` for the current chunk only
- after the chunk is written, control returns to the small model and routing is re-evaluated at the next boundary

## Goal

Move LEAP from post-hoc rollback to pre-decode chunk routing.

The main question is:

> Given the current prefix state, should the current chunk be decoded by the small model or the large model?

## First implementation scope

Use a small MATH500 subset first.

1. Define candidate chunk boundaries.
2. Build offline labels for current-chunk decode choice.
3. Train a lightweight router.

## Mixed-utility training flow

When decisive-only labels are too sparse, we can convert labeled JSONL into a weighted `.pt`
dataset where:

- `utility_label = 2` stays as a high-confidence positive (`LLM`)
- `utility_label = 0` stays as a high-confidence negative (`SLM`)
- `utility_label = 1` is retained as a low-weight gray sample

Prepare the weighted dataset:

```bash
python -m experimental.chunk_decode_router.prepare_mixed_training_dataset \
  --input-path experimental/chunk_decode_router/math500_test100_decode_choice_labeled_full.jsonl \
  --output-path experimental/chunk_decode_router/math500_test100_decode_choice_mixed.pt \
  --mode mixed_utility \
  --gray-weight 0.35 \
  --positive-weight 1.0 \
  --negative-weight 1.0
```

Then train with the existing probe trainer. It now respects a top-level `sample_weight`
field when present:

```bash
python -m core_package.probes.train_probe_artifact_torch \
  --label-path experimental/chunk_decode_router/math500_test100_decode_choice_mixed.pt \
  --output-path result/artifacts/chunk_decode_router_probe_math500_mixed.pt \
  --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob \
  --label-key label \
  --pos-weight 1.0
```

## Preference-core training flow (Option C)

When utility-aware mixing remains unstable, we can switch to a lighter-weight
pairwise-preference view of the same rollout data:

- `utility_label = 2` -> `LLM preferred`
- `utility_label = 0` -> `SLM preferred`
- `utility_label = 1` -> excluded from the first-pass preference core

Prepare the preference-core dataset:

```bash
python -m experimental.chunk_decode_router.prepare_mixed_training_dataset \
  --input-path experimental/chunk_decode_router/math500_test100_decode_choice_risk_full.jsonl \
  --output-path experimental/chunk_decode_router/math500_test100_decode_choice_C_preference.pt \
  --mode preference_core \
  --positive-weight 1.0 \
  --negative-weight 1.0
```

Train the first-pass preference probe:

```bash
python -m core_package.probes.train_probe_artifact_torch \
  --label-path experimental/chunk_decode_router/math500_test100_decode_choice_C_preference.pt \
  --output-path result/artifacts/chunk_decode_router_probe_C_preference.pt \
  --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob \
  --label-key label \
  --pos-weight 1.0
```
4. Add an inference loop that routes the current chunk before decoding.

## Implemented first pass

The first code path in this folder is:

- `build_decode_choice_dataset.py`

It supports two stages:

1. Build candidate current-chunk routing samples from an existing `build_dataset.py` trajectory file.
2. Optionally run local dual-branch comparisons:
   - `SLM_chunk + SLM_rest`
   - `LLM_chunk + SLM_rest`

and emit a first-pass hard/soft label package for the current chunk.

The dataset now stores training-ready feature fields for the current candidate chunk:

- `boundary_hidden_state`
- `mean_hidden_state`
- `relative_position`
- `final_entropy`
- `final_top1_prob`
- `final_margin`
- `mean_entropy`

When rollout labeling is enabled, it also writes:

- top-level `label` (`1` means `LLM`, `0` means `SLM`)
- top-level `utility_label`
- full comparison metadata under `comparison`

## Example usage

Build candidate rows only:

```bash
python -m experimental.chunk_decode_router.build_decode_choice_dataset \
  --trajectory-path dataset/math500_test_15b_hidden_states_hf_t2048.pt \
  --output-path experimental/chunk_decode_router/math500_test100_decode_choice_candidates.jsonl \
  --dataset-name math500 \
  --answer-type math500_qwen_boxed \
  --num-questions 100 \
  --candidate-policy uniform_plus_ends \
  --candidate-count 4
```

Build candidates and run current-chunk rollout comparisons:

```bash
python -m experimental.chunk_decode_router.build_decode_choice_dataset \
  --trajectory-path dataset/math500_test_15b_hidden_states_hf_t2048.pt \
  --output-path experimental/chunk_decode_router/math500_test100_decode_choice_labeled.jsonl \
  --dataset-name math500 \
  --answer-type math500_qwen_boxed \
  --num-questions 100 \
  --candidate-policy uniform_plus_ends \
  --candidate-count 4 \
  --label-with-rollouts \
  --small-model-path models/Qwen2.5-Math-1.5B-Instruct \
  --large-model-path models/Qwen2.5-32B \
  --large-backend vllm \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key token-abc123 \
  --vllm-model-name Qwen2.5-32B
```

## Helper scripts

- `run_vllm_32b.sh`
  starts a `Qwen2.5-32B` vLLM server for decode-choice labeling and keeps the same large-model setting as the current LEAP mainline.
- `label_math500_decode_choice_test10.sh`
  runs a first 10-question labeled decode-choice pass on MATH500 using:
  - `Qwen2.5-Math-1.5B-Instruct` for `SLM_chunk + SLM_rest`
  - `Qwen2.5-32B` for `LLM_chunk`
  - `vLLM` as the large-model backend
- `analyze_decode_choice_labels.py`
  summarizes candidate density, hard-label counts, utility-label counts, and optional `LLM` examples from a labeled decode-choice jsonl.

Inspect a labeled subset:

```bash
python -m experimental.chunk_decode_router.analyze_decode_choice_labels \
  --input-path experimental/chunk_decode_router/math500_test10_decode_choice_labeled.jsonl \
  --show-examples 3
```

Train a first-pass decode router with the existing probe trainer:

```bash
python -m core_package.probes.train_probe_artifact_torch \
  --label-path experimental/chunk_decode_router/math500_test10_decode_choice_labeled.pt \
  --output-path result/artifacts/chunk_decode_router_probe_test10.pt \
  --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob \
  --label-key label
```

## Files in this workspace

- `PLAN.md`
  implementation plan and milestones
- `label_schema.md`
  label design for current-chunk `SLM` vs `LLM` decode
- `notes.md`
  experiment notes and observations
- `build_decode_choice_dataset.py`
  offline candidate builder and first-pass decode-choice labeler

This line stays isolated from the current mainline until we are happy with the data contract and routing behavior.
