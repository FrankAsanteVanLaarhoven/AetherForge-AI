# v2.18 Phase B — Claim Boundary

## Verdict

**DIRECTIONAL.** Phase B shows directional improvement but does not exceed the promotion gate.

## What this measures

- Encoder under test: a code-pretrained encoder (768d) vs the protected
  code-aware MiniLM dense baseline (384d, mean 16.3/32).
- Clean 32-task benchmark, best-of-3, three runs per mode.
- Hybrid stage-1 shortlist is code-aware MiniLM dense (not TF-IDF); stage-2 reranks
  with the Phase B encoder.

## Promotion gate (strict)

- Minimum candidate: mean > 18.3/32 across three runs.
- Strong result: ≥22/32 on a run. Stronger: converts ≥1 stable-fail task to
  stable-pass without broad family-level regression.
- A single high run is not enough. A 28-task improvement alone is not enough.
- Single-family gain with other-family regression = family-specific, not promotion.

## Not claimed (regardless of result)

- No SWE-bench capability.
- No production-grade reliability.
- No AGI, quantum reasoning, or general superiority over other systems.
- No claim that dense retrieval beats true TF-IDF (the baseline is already dense).
- Results are bounded to the 32-task families tested at n=32, best-of-3.
- No AI/tool/vendor attribution.

## Next direction (if not promoted)

A larger or different encoder failing here does **not** mean dense retrieval
fails globally. The likely bottleneck is the memory-record format: records are
long ReAct trajectories mixing instruction, critique, tool calls, observations,
errors, and recovery logic. Recommended next direction: operation-aware metadata
or shorter memory summaries, evaluated against this same stabilised baseline.
