# v2.19b — Family-Targeted Memory Coverage Summary

**Baseline (Phase A, code-aware MiniLM dense):** mean 16.3/32 = 51.0%, std 0.94, range 15–17.  
**Promotion gate:** mean > 18.3/32 across three runs.

Tests the v2.19 conclusion that the bottleneck is memory COVERAGE. Adds 16 verified
same-family-different-task repair records (interval/tree/rle/dict; contamination-guarded,
names disjoint from the 32 benchmark tasks), encoder held fixed.

## Results (clean 32-task benchmark, best-of-3)

| Mode | run1 | run2 | run3 | mean | std | range | Δ vs baseline | gate |
|---|---|---|---|---|---|---|---|---|
| Baseline MiniLM dense | 17 | 15 | 17 | 16.3 | 0.94 | 15–17 | — | — |
| v219b structured-dense (combined pool) | 19 | 20 | 19 | 19.3 | 0.47 | 19–20 | +3.0 | PASS |

## Verdict

**PROMOTED** — Structured memory records are promoted as a retrieval candidate for further evaluation, not as production evidence and not as SWE-bench evidence.

- Single-run ≥22/32 (strong): none.
- Stable-fail → stable-pass conversions: `v210_interval_intersection`.
  - of which retrieve a NEW family-targeted record (coverage-attributable): `v210_interval_intersection`.
- Stable-pass → stable-fail hard regressions: none.
- Tasks with changed pass count: 12.

### Interpretation

- Variance dropped sharply (std 0.47 vs baseline 0.94 and v2.19's
  2.49); the lift is stable across runs, not a single lucky draw — the strongest retrieval
  result in the arc so far.
- It supports the v2.19 conclusion that COVERAGE was a real bottleneck: the one hard
  conversion is in the interval family that received coverage and is coverage-attributable
  (`interval_intersection` now retrieves related interval records and writes its own correct
  two-pointer solution — related memory, not its answer, so no leakage).
- But coverage is NECESSARY-NOT-SUFFICIENT: `interval_union` and every tree stable-fail
  (`tree_from_list`, `tree_max_path_sum`, `tree_serialize`, `tree_width`) still fail despite
  now retrieving relevant same-family records — relevant memory helps some tasks, not all.
- Part of the mean lift is flip→pass stabilisation, including `find_peak_element` in the
  UNCOVERED search family, so a portion is sampling-side, not coverage. Treat as a candidate
  for confirmation, not a robust production win.

See `comparison.csv`, `per_task_matrix.csv`, `per_family_breakdown.md`,
`retrieval_trace.md`, and `claim_boundary.md`.
