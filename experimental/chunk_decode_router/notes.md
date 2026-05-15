# Notes

## Why this line exists

The current clean055 trace suggests the main limitation is not random false alarms, but late routing:

- some wrong questions are rescued
- almost no correct questions are harmed
- many triggered cases are still not repaired

This motivates moving the decision earlier:

- from "observe after the chunk is generated"
- to "route before the current chunk is decoded"

## Key risk

If we recompute full prefix prefill at every boundary, routing overhead may erase the benefit.

So the preferred implementation should reuse online prefix state whenever possible, instead of recomputing from scratch.

## First engineering compromise

The first offline dataset pass does not change the online scheduler yet.

Instead, it:

1. reuses existing `build_dataset` trajectories to define candidate current-chunk boundaries
2. compares local alternatives offline
3. learns whether the current chunk is worth assigning to `LLM`

This lets us stabilize the supervision target before rewriting the runtime routing loop.
