# v2.21b — Tree-Width Planning Ablation Summary

Execution-plan prompt with the worked example REMOVED (same contract, same v2.19c
retrieval). Tests whether the v2.21 tree_width conversion came from the plan structure
or the example.

## tree_width (the ablation target)

- v2.21 full prompt: **6/6**
- v2.21b ablation (no example): **9/9** (rate 100%) [subset 6/6, full-32 3/3]

## All tree stable-fails

| Task | Baseline | v2.21 full | v2.21b subset | v2.21b full-32 |
|---|---|---|---|---|
| v210_tree_from_list | 0/3 | 1/6 | 1/6 | 1/3 |
| v210_tree_max_path_sum | 0/3 | 0/6 | 0/6 | 0/3 |
| v210_tree_serialize | 0/3 | 2/6 | 1/6 | 1/3 |
| v210_tree_width | 0/3 | 6/6 | 6/6 | 3/3 |

Aggregate full-32 under ablation: [20, 17, 19] → mean 18.7/32 (baseline 16.3, v2.21 18.3).
Hard regressions vs baseline: `v210_flatten_dict`.

## Plan-adherence guard metrics

Over 120 trajectories: PLAN 100%, base-case 82%, combine 100%, minimal-test 97%, repair 27%.

## Verdict

**SURVIVES** — The tree_width conversion SURVIVES removal of the worked example, strengthening the claim that structured execution planning improved a reasoning-control-bound tree task.

See `comparison.csv`, `tree_stablefails.csv`, `guard_metrics.csv`, `claim_boundary.md`.
