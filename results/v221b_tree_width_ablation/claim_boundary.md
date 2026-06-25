# v2.21b — Claim Boundary

## Verdict

**SURVIVES.** The tree_width conversion SURVIVES removal of the worked example, strengthening the claim that structured execution planning improved a reasoning-control-bound tree task.

## What this measures

- The v2.21 execution-plan prompt with ONLY the worked example removed (contract,
  retrieval, model, and all else identical) — a clean single-variable ablation.
- Decision metric: tree_width pass rate under ablation vs the v2.21 6/6 result.

## Not claimed

- No SWE-bench success; no production reliability; no model-weight change.
- No frontier-model superiority; general tree reasoning is NOT solved.
- The other tree stable-fails remain capability-bound (fail despite plan adherence).
- Bounded to the 32-task benchmark, best-of-3.
- No AI/tool/vendor attribution.
