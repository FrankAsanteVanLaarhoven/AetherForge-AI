# v2.19c — Claim Boundary

## Confirmation

**CONFIRMED.** v2.19b CONFIRMED: mean 19.8/32 > gate 18.3 over 10 seeds.

## Targeted coverage

**COVERAGE_PARTIAL.** SPLIT result. Targeted coverage CONVERTS interval_union (0/3 -> 3/3) via the interval list-building records — coverage transfers for that pattern. But NO tree stable-fail converts despite targeted tree records, so the persistent tree failures are reasoning/control-bound, not coverage-bound. The expanded aggregate (18.3) sits within the confirmation variance band, so adding records did not clearly raise the aggregate. Adopt the interval coverage selectively; do not promote the expanded pool wholesale; treat tree as reasoning-bound and the next target for control/curriculum work, not more memory.

## Contamination guard

- All v2.19c targeted records are distinct algorithms; function names asserted
  disjoint from the 32 benchmark callables; each solution execution-verified.
- The retrieval trace confirms persistent-failure tasks retrieve RELATED records
  (e.g. interval_union → interval_gaps/complement; tree_width → tree_count_at_depth),
  never their own answer — the benchmark stays independent.

## Not claimed

- No SWE-bench success; no production reliability; no frontier-model superiority.
- No model-weight change; no code-agent SOTA.
- Conversions credited to coverage only when the converted task retrieves an added
  same-family record (see retrieval trace); otherwise flip-task variance.
- Bounded to the 32-task families at n=32, best-of-3.
- No AI/tool/vendor attribution.
