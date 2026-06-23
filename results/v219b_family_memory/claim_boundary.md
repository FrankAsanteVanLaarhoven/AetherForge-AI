# v2.19b — Claim Boundary

## Verdict

**PROMOTED.** Structured memory records are promoted as a retrieval candidate for further evaluation, not as production evidence and not as SWE-bench evidence.

## What this measures

- Whether adding same-family-DIFFERENT-task verified repair memory (interval/tree/
  rle/dict) lets the benchmark tasks retrieve family-relevant guidance and convert.
- Encoder held FIXED (baseline MiniLM); same structured retrieval as v2.19; only the
  memory pool grew by 16 contamination-guarded records.

## Contamination guard

- The 16 authored records are distinct algorithms with function names disjoint from
  all 32 benchmark callables (asserted in build_v219b_family_records.py).
- Each authored solution is verified by execution before inclusion.
- The retrieval trace confirms benchmark tasks retrieve RELATED family records, not
  their own answers — so the benchmark remains independent.

## Not claimed (regardless of result)

- No SWE-bench capability or success; no production-grade reliability.
- No frontier-model superiority; no AGI or quantum-reasoning claims.
- Results bounded to the 32-task families at n=32, best-of-3.
- Conversions are credited to coverage only when the converted task retrieves a NEW
  same-family record (see summary); otherwise they are flip-task variance.
- No AI/tool/vendor attribution.
