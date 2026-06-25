# v2.21 — Claim Boundary

## Verdict

**CANDIDATE.** ForgeReasoningCore-style execution planning is promoted as a reasoning-control candidate for further evaluation.

## What this measures

- An EXECUTION-PLAN prompt (plan → code → test → repair → final) layered on the SAME
  v2.19c expanded retrieval, isolating the prompt/control variable from retrieval.
- Decision metric is tree stable-fail conversions, not aggregate mean.
- Curriculum records (data/v221_reasoning_curriculum.jsonl) are tree-family
  NON-benchmark, execution-verified, contamination-guarded.

## Not claimed

- No SWE-bench success; no production reliability; no frontier-model superiority.
- No model-weight change.
- Aggregate-mean movement is not a promotion basis when tree failures do not move.
- Bounded to the 32-task benchmark, best-of-3.
- No AI/tool/vendor attribution.
