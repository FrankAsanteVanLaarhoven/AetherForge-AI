# v2.19c — Confirmation + Targeted Coverage Summary

**Baseline:** mean 16.3/32, std 0.94. **Gate:** mean > 18.3. **v2.19b prior:** 19.3/32.

## Phase 1 — Confirmation (v2.19b combined pool, extended seeds)

Runs (10): [22, 22, 17, 19, 17, 24, 19, 19, 20, 19]  
mean **19.8/32**, std 2.14, range 17–24  
runs above gate (>18.3): 8/10  
runs ≥22: 3/10  
interval_intersection retained pass-rate: 10/10  

**CONFIRMED** — v2.19b CONFIRMED: mean 19.8/32 > gate 18.3 over 10 seeds.

## Phase 3 — Targeted Coverage Expansion (pool 99+16+6=121)

| Cohort | runs | mean | std | range | Δ vs baseline |
|---|---|---|---|---|---|
| Baseline | 3 | 16.3 | 0.94 | 15–17 | — |
| Expanded dense | 3 | 18.3 | 0.94 | 17–19 | +2.0 |

- tree family: baseline 3.00 → expanded 3.00 (+0.00).
- interval_union: baseline 0/3 → expanded 3/3.
- Expanded stable-fail→pass conversions: `v210_interval_intersection`, `v210_interval_union`.
- Expanded hard regressions: none.

**COVERAGE_PARTIAL** — SPLIT result. Targeted coverage CONVERTS interval_union (0/3 -> 3/3) via the interval list-building records — coverage transfers for that pattern. But NO tree stable-fail converts despite targeted tree records, so the persistent tree failures are reasoning/control-bound, not coverage-bound. The expanded aggregate (18.3) sits within the confirmation variance band, so adding records did not clearly raise the aggregate. Adopt the interval coverage selectively; do not promote the expanded pool wholesale; treat tree as reasoning-bound and the next target for control/curriculum work, not more memory.

See `confirmation.csv`, `per_task_matrix.csv`, `per_family_breakdown.md`,
`retrieval_trace.md`, `claim_boundary.md`.
