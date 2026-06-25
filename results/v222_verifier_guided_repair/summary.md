# v2.22 — Verifier-Guided Multi-Step Repair Summary

**Baseline:** 16.3/32. **v2.21b (plan, no repair):** the 3 capability-bound tasks at 0–1/6. Judged by capability-bound conversions, not aggregate mean.

## Capability-bound tree tasks (the decision metric)

| Task | Baseline | v2.21b plan | v2.22 repair subset | v2.22 repair full-32 | Converted | Repair-attributable |
|---|---|---|---|---|---|---|
| v210_tree_serialize | 0/3 | 1/6 | 1/3 | 0/3 | no | no |
| v210_tree_from_list | 0/3 | 1/6 | 2/3 | 1/3 | no | no |
| v210_tree_max_path_sum | 0/3 | 0/6 | 1/3 | 1/3 | no | no |

**Conversions: 0** (repair-attributable: 0) .
Full-32 hard regressions vs baseline: none.

## Repair dynamics

- Trajectories with a VERIFIER signal: 76/105 (avg 1.07 blocks/trajectory).
- Repaired to PASS (had a VERIFIER signal then passed): 61/105.
- Hit repair budget: 11/105.

## Aggregate (secondary — not a promotion basis)

- v2.22 full-32: [23, 23, 20] → mean 22.0/32 (baseline 16.3).

## Verdict

**NO_CONVERSION** — Bounded verifier-guided repair did not overcome the capability-bound tree failures; the repair traces localize the failure for a future targeted fine-tune.

## Notable secondary finding (NOT the promotion basis)

- The repair loop is broadly effective: **61/105 trajectories repaired to PASS** (had a VERIFIER signal then passed), lifting full-32 to **mean 22.0/32** ([23, 23, 20]) — the highest aggregate in the arc (+5.7 vs baseline 16.3, vs v2.21 18.3), with no hard regressions.
- BUT this is not promoted and not the milestone's target: the 3 capability-bound tree tasks still do NOT stably convert (flip up, not 3/3), so they remain capability-bound and are the v2.23 targeted-fine-tune target.
- CONFOUND: v2.22 adds BOTH a precise VERIFIER signal AND a disciplined repair budget/no-repeat vs v2.21. The aggregate lift could be either; a v2.22b ablation (same budget + no-repeat but RAW stderr instead of the VERIFIER signal) would attribute it — mirroring the v2.21b worked-example ablation.

See `comparison.csv`, `capbound.csv`, `repair_dynamics.csv`, `claim_boundary.md`.
