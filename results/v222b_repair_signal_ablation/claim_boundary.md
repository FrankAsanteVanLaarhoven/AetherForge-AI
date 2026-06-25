# v2.22b — Claim Boundary

## Verdict

**SIGNAL_ATTRIBUTED.** The structured VERIFIER scores 5.3 tasks above raw stderr (verifier 22.0 vs raw 16.7, which sits at baseline); the signal FORMAT — distilling the failure into a labeled, actionable block — drives the v2.22 lift, not the repair discipline. The 1.5B model acts on the distilled signal but not on a raw traceback.

## What this measures

- v2.22b changes ONE variable vs v2.22: the failed-execution OBSERVATION is RAW stderr
  instead of the distilled VERIFIER block. Budget, no-repeat, diagnostic-assert
  contract, retrieval, and model are identical.
- Attributes the v2.22 aggregate lift to signal format vs repair discipline.

## Not claimed

- No SWE-bench success; no production reliability; no model-weight change.
- No frontier-model superiority; the 3 capability-bound tree tasks remain unsolved.
- Bounded to the 32-task benchmark, best-of-3.
- No AI/tool/vendor attribution.
