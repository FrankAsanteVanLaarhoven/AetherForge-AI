# v2.18 Claim Boundary

## What Phase A will establish

1. **Stable baseline:** mean TF-IDF score across 3 runs ± measured range.
2. **Noise floor:** the number of tasks that flip between runs without any retrieval change.
3. **Stable-pass tasks:** tasks that pass reliably (all 3 runs) — these are not subject to noise.
4. **Stable-fail tasks:** tasks that fail reliably — likely model capability ceiling.
5. **Flip tasks:** tasks where the result varies — these require multi-run averaging.

## What Phase B can prove

| Claim | Evidence required |
|---|---|
| Code-dense retrieval beats TF-IDF | Beat stabilised TF-IDF mean by > noise floor, ≥22/32 |
| Code-dense improves specific family | Family score improvement outside noise, confirmed across ≥2 runs |
| Code-hybrid beats pure dense | Hybrid > dense on 32-task by > noise floor |
| MiniLM null result is embedder-specific | Code-dense succeeds where MiniLM tied |

## What Phase B cannot prove

- Results outside the 32-task task families tested
- Superiority over larger models or cloud APIs
- Benefit on multi-file repository tasks (SWE-bench scope)
- Statistical significance at n=32 with best-of-3 sampling

## Promotion gate (strict)

Dense or hybrid is promoted to champion only if ALL of the following hold:
1. Clean 32-task score beats the stabilised TF-IDF mean by > measured noise floor
2. Score reaches ≥22/32 = 68.8% (strong evidence threshold)
3. No family-level regression worse than the noise floor
4. Result is confirmed to use only `memory/index_adapted` records (99 verified, no repair)

A family-specific improvement (e.g. +2 in interval, 0 elsewhere) is recorded as
"family-specific improvement" not a global champion promotion.

## Forbidden claims (regardless of results)

- Do not claim SWE-bench success.
- Do not claim production-grade reliability.
- Do not claim AGI, quantum reasoning, or general superiority.
- Do not claim dense retrieval outperforms TF-IDF unless clean 32-task evidence supports it.
- Do not attribute results to any AI system, vendor, or tool.
