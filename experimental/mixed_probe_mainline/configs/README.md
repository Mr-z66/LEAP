# Configs

Keep fixed configuration files for the mixed-domain mainline here.

Recommended first config:

- split seed
- calibration/test sizes per dataset
- small/large/judge model paths
- scheduler thresholds
- output naming conventions

`autodl_hf_kv_fixed4.env` is the canonical runnable configuration for the
HF-only mixed-probe fixed4 experiment.  It points to persistent AutoDL data,
the chunk-level balanced-5:1 probe, and the uniform 1.5B -> 7B model pairing.
