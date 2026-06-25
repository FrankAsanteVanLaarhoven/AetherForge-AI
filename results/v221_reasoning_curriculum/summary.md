# v2.21 — ForgeReasoningCore Execution-Plan Summary

**Baseline:** 16.3/32. **v2.19c expanded control (no plan):** tree stable-fails 0/3. Judged by tree conversions, not aggregate mean.

## Tree stable-fail outcomes (the decision metric)

| Task | Baseline | v2.19c control | v2.21 subset | v2.21 full-32 | Converted |
|---|---|---|---|---|---|
| v210_tree_from_list | 0/3 | 0/3 | 1/3 | 0/3 | no |
| v210_tree_max_path_sum | 0/3 | 0/3 | 0/3 | 0/3 | no |
| v210_tree_serialize | 0/3 | 0/3 | 1/3 | 1/3 | no |
| v210_tree_width | 0/3 | 0/3 | 3/3 | 3/3 | yes |

**Tree stable-fail conversions: 1** (`v210_tree_width`).
Full-32 hard regressions vs baseline: none.

## Aggregate (secondary — not a promotion basis if tree does not move)

- v2.21 full-32: [17, 21, 17] → mean 18.3/32 (baseline 16.3).

## Guard metrics (did the model follow the plan structure?)

Over 108 trajectories: PLAN emitted 100%, base-case 100%, combine 100%, minimal-test 96%, repair 26%.

## Verdict

**CANDIDATE** — ForgeReasoningCore-style execution planning is promoted as a reasoning-control candidate for further evaluation.


## Interpretation

- The model FULLY adopted the plan structure (PLAN 100%, base-case 100%, combine 100%). So the 3 tree tasks that still fail (serialize, from_list, max_path_sum) fail DESPITE correct planning — they are capability-bound (the 1.5B model cannot write the specific string-building / BST-reconstruction / any-path-DP logic), not control-bound.
- `tree_width` (level counting) is the one conversion. CAVEAT: the execution-plan prompt's worked example (`tree_count_at_depth`) demonstrates a RELATED level-counting recursion, and retrieval surfaces `tree_level_counts`/`tree_count_at_depth`, so this conversion may be partly example/coverage-aided rather than pure abstract planning. An ablation (plan prompt WITHOUT the tree worked example) would disentangle this; deferred.
- Net: 'tree' splits further — `tree_width` was reasoning/control-bound (planning fixes it); the rest are capability-bound. Repair fired in 26% of trajectories but did not rescue the capability-bound tasks.

See `comparison.csv`, `tree_stablefails.csv`, `guard_metrics.csv`, `claim_boundary.md`.
