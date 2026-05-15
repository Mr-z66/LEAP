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
