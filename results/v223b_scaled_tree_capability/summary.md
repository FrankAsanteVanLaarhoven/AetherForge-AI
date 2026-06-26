# v2.23b — Scaled Tree Capability Adapter Summary

Scales the v2.23 pilot: data 49→94 (27 distinct contamination-guarded tasks), steps 50→150, same low LR. Fresh LoRA on the merged champion (champion untouched), evaluated under the v2.22 verifier mode + v2.19c retrieval. Tests data-limited vs capability ceiling.

## Hard tree tasks (the decision metric)

| Task | Baseline | Champion+ver | Pilot v2.23 | Scaled v2.23b | Scaled NO-ver | Converted |
|---|---|---|---|---|---|---|
| v210_tree_serialize | 0/3 | 1/3 | 1/3 | 2/3 | 0/3 | no |
| v210_tree_from_list | 0/3 | 2/3 | 1/3 | 0/3 | 0/3 | no |
| v210_tree_max_path_sum | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 | no |

**Hard-tree conversions (scaled+verifier): 0** (none).
Persist without verifier (weight-level): 0.

## Full-32 (regression gate)

- Scaled adapter+verifier: [20, 24, 18] → mean 20.7/32 (std 2.49; champion 22.0; baseline 16.3).
- Hard regressions: none.
- Within v2.22 band (≥20.0): yes.

## Verdict

**CEILING** — Scaling the targeted training (~2x data, 3x steps) still did not convert any hard tree task. Combined with the v2.23 pilot, this indicates the residual failures are at or near the 1.5B capability ceiling for these patterns under the contamination-guarded protocol, not simply data-limited.

### Data-limited vs ceiling

- The pilot (49 ex / 50 steps) and the scaled run (94 ex / 150 steps) both yield ZERO hard-task conversions. Doubling data and tripling steps moved nothing → the evidence favors a capability ceiling over a data limit for these three patterns at this model size, under contamination control.
- Scaling also began to COST aggregate performance: full-32 fell from the pilot's 22.7 to 20.7 with higher variance (a low run near baseline), and tree_from_list/max_path_sum did not improve. No hard stable-pass regression occurred, so it stayed within the band — but the dip is the project's familiar over-training drift surfacing even under the controlled protocol. More targeted training is not the fix; it trades general performance without cracking the hard tasks.

See `comparison.csv`, `hardtree.csv`, `claim_boundary.md`.
