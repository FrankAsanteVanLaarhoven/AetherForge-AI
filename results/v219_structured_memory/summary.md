# v2.19 — Structured Memory Records + Query Reranking Summary

**Baseline (Phase A, code-aware MiniLM dense, 384d):** mean 16.3/32 = 51.0%, std 0.94, range 15–17.  
**Promotion gate:** mean > 18.3/32 across three runs.

Encoder held fixed (baseline code-aware MiniLM); only record structure, multi-view
query construction, and deterministic reranking change.

## Results (clean 32-task benchmark, best-of-3)

| Mode | run1 | run2 | run3 | mean | std | range | Δ vs baseline | gate |
|---|---|---|---|---|---|---|---|---|
| Baseline MiniLM dense | 17 | 15 | 17 | 16.3 | 0.94 | 15–17 | — | — |
| structured_dense | 20 | 16 | 18 | 18.0 | 1.63 | 16–20 | +1.7 | no |
| structured_hybrid | 22 | 16 | 20 | 19.3 | 2.49 | 16–22 | +3.0 | PASS |

## Verdict

**PROVISIONAL** — Structured-hybrid meets the minimum mean gate but shows no stable-fail -> stable-pass conversion in that mode and high run-to-run variance; the retrieval trace shows surfaced memory is largely family-irrelevant, so the mean lift is consistent with flip-task sampling variance. Recorded as a PROVISIONAL candidate requiring confirmation runs, not a promotion.

Best mode: **structured_hybrid** (mean 19.3/32, Δ +3.0).

- Single-run ≥22/32 (strong): structured_hybrid=[22].
- Stable-fail → stable-pass conversions by mode: structured_dense=1, structured_hybrid=0 (tasks: `v210_kth_smallest_matrix`).
- Stable-pass → stable-fail hard regressions: none.
- Tasks with changed pass count: 19.

### Mechanistic read (why the gate result must be read with caution)

`retrieval_trace.md` re-runs the deterministic retriever for every changed task. For
the interval/rle/tree gains driving the structured-hybrid mean (e.g. `range_summary`,
`meeting_rooms`, `interval_intersection`), the surfaced memory is family-IRRELEVANT
(`unique_sorted`, `two_sum`, `merge_sorted`) — the 99-record pool contains essentially
no verified interval/tree/rle repairs. So those pass-count movements are not
mechanistically attributable to retrieval; they are consistent with best-of-3 flip-task
variance. The only retrieval-attributable hard conversion is `kth_smallest_matrix`
(0→3/3) in structured-DENSE, which correctly surfaced the lone `transpose(matrix)`
record — but structured-dense does not clear the gate, and hybrid scores that task 0/3.

See `comparison.csv`, `per_task_matrix.csv`, `per_family_breakdown.md`,
`retrieval_trace.md`, and `claim_boundary.md`.
