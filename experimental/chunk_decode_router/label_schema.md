# Label Schema Draft

## Task

At chunk boundary `t`, decide who should decode the current chunk:

- `SLM`
- `LLM`

## Core formulation

Given prefix `p_t`, compare:

1. `p_t + chunk_t^S + rest^S`
2. `p_t + chunk_t^L + rest^S`

Where:

- `chunk_t^S` is the current chunk decoded by the small model
- `chunk_t^L` is the current chunk decoded by the large model
- `rest^S` is the continuation decoded by the small model

This keeps the intervention local: only the current chunk changes author.

## Candidate label styles

### Option A: hard label

- `LLM` if the `LLM`-decoded current chunk leads to a better final outcome
- otherwise `SLM`

### Option B: three-way utility label

- `2`: strong benefit from `LLM`
- `1`: weak or uncertain benefit
- `0`: `SLM` is sufficient or `LLM` is not worth the cost

### Option C: pairwise preference

Store whether `LLM` is preferred over `SLM` at the current chunk boundary, without forcing a hard class during dataset construction.

## Recommendation for first pass

Start with:

- pairwise comparison metadata
- plus a coarse three-way utility label

This keeps the dataset reusable if we later decide between classification, ranking, or utility regression.

## First-pass concrete mapping

For the initial implementation:

- `label = "LLM"`
  - when `LLM_chunk + SLM_rest` is correct and `SLM_chunk + SLM_rest` is incorrect
- `label = "SLM"`
  - otherwise

Coarse utility labels:

- `2`
  - `LLM` clearly improves correctness
- `1`
  - both branches agree in correctness outcome, or the gain is ambiguous
- `0`
  - `SLM` is clearly better

This is intentionally conservative for the first dataset pass.
