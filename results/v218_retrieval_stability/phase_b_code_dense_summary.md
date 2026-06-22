# v2.18 Phase B — Code-Specialised Dense Retrieval Summary

**Baseline (Phase A, code-aware MiniLM dense, 384d):** mean 16.3/32 = 51.0%, range 15–17.  
**Promotion gate:** mean > 18.3/32 across 3 runs (baseline mean + 2 noise floor).

## Results (clean 32-task benchmark, best-of-3)

| Mode | run1 | run2 | run3 | mean | range | Δ vs baseline | gate |
|---|---|---|---|---|---|---|---|
| Baseline MiniLM dense | 17 | 15 | 17 | 16.3 | 15–17 | — | — |
| Code-dense | 15 | 20 | 20 | 18.3 | 15–20 | +2.0 | no |
| Code-hybrid | 19 | 15 | 19 | 17.7 | 15–19 | +1.3 | no |

## Verdict

**DIRECTIONAL** — Phase B shows directional improvement but does not exceed the promotion gate.

Best Phase B mode: **code-dense** (mean 18.3/32, Δ +2.0 vs baseline).

- Single-run ≥22/32 (strong): none.
- Stable-fail → stable-pass conversions: none.
- Stable-pass → stable-fail regressions: none.

See `phase_b_code_dense_comparison.csv`, `phase_b_per_task_matrix.csv`,
`phase_b_per_family_breakdown.md`, and `phase_b_claim_boundary.md`.
