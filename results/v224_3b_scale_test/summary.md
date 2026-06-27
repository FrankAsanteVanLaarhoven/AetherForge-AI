# v2.24 — 3B-Scale Capability Ceiling Test Summary

Tests whether ~2x model scale (1.5B → Qwen2.5-Coder-3B) moves the residual capability ceiling on the 3 hard tree tasks. Same v2.22 verifier mode + v2.19c retrieval. The 3B base and adapter are separate; the frozen 1.5B champion (23/28=82.1%) is untouched.

## Hard tree tasks (the decision metric)

| Task | 1.5B champion+ver | 3B base+ver | 3B adapter+ver | 3B adapter NO-ver |
|---|---|---|---|---|
| v210_tree_serialize | 1/3 | 1/3 | 0/3 | 2/3 |
| v210_tree_from_list | 2/3 | 3/3 | 3/3 | 2/3 |
| v210_tree_max_path_sum | 1/3 | 2/3 | 3/3 | 0/3 |

**Hard-task conversions — 3B base: 1 (`v210_tree_from_list`); 3B adapter: 2 (`v210_tree_from_list`, `v210_tree_max_path_sum`).**

## Full-32 (context; 3B is a different model, not directly comparable to the 16.3 baseline)

- 1.5B champion+verifier: mean 22.0 ([23,23,20]).
- 3B base+verifier: [26, 26, 27] → mean 26.3 (std 0.47).
- 3B adapter+verifier: [29, 28, 30] → mean 29.0
- 3B adapter regressions vs 3B base (3/3 → 0/3): none.

## Verdict

**SCALE_HELPS** — The 3B BASE alone converts 1 hard task(s) the 1.5B champion could not — the residual capability wall is SCALE-DEPENDENT, not absolute. Targeted training then converts a further 1 task(s) (`v210_tree_max_path_sum`) on top of the 3B base — the SAME contamination-guarded recipe that did nothing at 1.5B (v2.23/v2.23b). Scale did not just solve a task; it made the model trainable for these patterns.

### Interpretation

- **Scale is the primary lever.** The 3B base (no champion fine-tune, no targeted training) converts `tree_from_list` (3/3) and lifts full-32 to 26.3 vs the 1.5B champion's 22.0. The v2.23/v2.23b 'capability ceiling' was specific to the 1.5B class, not the tasks.
- **Targeted training compounds with scale.** The 3B adapter converts a SECOND hard task (`tree_max_path_sum`) and reaches full-32 29.0 (no regression vs the 3B base) — whereas the identical contamination-guarded recipe converted nothing at 1.5B. Scale made the model TRAINABLE for these patterns; targeted training only 'took' once the base was capable enough.
- `tree_serialize` (exact string format) is the lone holdout (0–2/3) — the hardest of the three even at 3B.
- Next scientific step: a larger jump (7B, deferred for VRAM/engineering) to test whether the last task and further headroom follow the same scale trend.

See `comparison.csv`, `hardtree.csv`, `claim_boundary.md`.
