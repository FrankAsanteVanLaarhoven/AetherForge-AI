# v2.19 — Claim Boundary

## Verdict

**PROVISIONAL.** Structured-hybrid meets the minimum mean gate but shows no stable-fail -> stable-pass conversion in that mode and high run-to-run variance; the retrieval trace shows surfaced memory is largely family-irrelevant, so the mean lift is consistent with flip-task sampling variance. Recorded as a PROVISIONAL candidate requiring confirmation runs, not a promotion.

## What this measures

- Structured memory records (deterministically derived: family, signature, failure
  mode, cues, minimal test, rationale) + multi-view query (instruction/family/
  signature) + deterministic composite reranking.
- Encoder held FIXED at the baseline code-aware MiniLM (384d), so any change is
  attributable to record structure and ranking, not encoder capacity.
- Same protected 99-record memory pool; injection format unchanged.
- Clean 32-task benchmark, best-of-3, three runs per mode.

## Promotion gate (strict)

- Minimum candidate: mean > 18.3/32 across three runs.
- Strong result: ≥22/32 on a run. Strongest: ≥1 stable-fail → stable-pass without
  broad family-level regression.
- A single high run is not enough. A 28-task improvement alone is not enough.

## Not claimed (regardless of result)

- No SWE-bench capability or success.
- No production-grade reliability.
- No frontier-model superiority; no AGI or quantum-reasoning claims.
- Results bounded to the 32-task families tested at n=32, best-of-3.
- No AI/tool/vendor attribution.

## Interpretation / next direction

The retrieval trace shows which records the structured reranker surfaces for the
changed tasks. Where same-family records do not exist in the 99-record pool, no
structuring or reranking can surface them — indicating the bottleneck is memory
*coverage* (the pool lacks family-relevant verified repairs for the benchmark
families), not only memory *format*. Recommended next direction: expand the
verified-repair memory with family-targeted records (interval/tree/rle/dict),
then re-run this structured-retrieval audit against the same stabilised baseline
before escalating to a heavier embedding backend.
