# Development Plan

## Phase 1: Data and labeling

Define a new sample unit:

- question id
- chunk boundary id
- current prefix text
- current prefix features
- current-chunk decode alternatives:
  - `SLM_chunk + SLM_rest`
  - `LLM_chunk + SLM_rest`

Deliverables:

- candidate chunk selection rule
- offline label generation script
- first labeled subset on MATH500

Current first-pass implementation target:

- script: `build_decode_choice_dataset.py`
- dataset: `MATH500 test100`
- candidate policy: `uniform_plus_ends`
- comparison mode:
  - `SLM_chunk + SLM_rest`
  - `LLM_chunk + SLM_rest`

## Phase 2: Router training

Train a router that predicts the current-chunk decode action:

- `SLM`
- `LLM`

Possible training targets:

- hard class label
- soft utility score
- pairwise preference

Start with the simplest stable target after inspecting the first labeled subset.

Current recommendation:

- store hard label
- store coarse utility label
- keep full pairwise comparison metadata for later relabeling

## Phase 3: Online decode loop

Replace rollback-style routing with chunk-level pre-decode routing:

1. stop at chunk boundary
2. read current prefix state
3. run router
4. assign current chunk to `SLM` or `LLM`
5. append decoded chunk
6. repeat at next boundary

## Phase 4: Evaluation

Primary comparison:

- current MATH500 clean055 baseline
- new chunk decode router

We should inspect:

- rescue count
- false-alarm count
- number of `LLM` chunks used
- average takeover cost
- failure modes
