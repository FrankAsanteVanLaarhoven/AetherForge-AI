# v2.35 — Solution-Body Generation After Tool-Call Recovery

Follows v2.34: tool-call emission was largely recovered (no_tool_call 26→1), but the frozen 32-task score stayed at 5/32 because the emitted execute_code bodies were incomplete, weakly asserted, or incorrect. v2.35 adds STRICT solution-body verification against each task's real benchmark assertions (never the model's `print('PASS')`) and classifies why bodies fail. This is NOT tool-call work and NOT repair training. No claim of model improvement, SOTA, SWE-bench, or production readiness.

## Results (frozen 32-task, v2.34 control + strict body verification)

- **Strict-verified pass: 5/32** (v2.34 baseline 5; champion 23).
- tool_call_rate **0.969**, execute_code_rate 0.188, no_tool_call 1, invalid_tool_json 6.
- Body classification: {'incomplete_no_def': 18, 'strict_pass': 5, 'assertion_failure': 7, 'no_tool_call': 1, 'fake_pass': 1}.
- fake_pass **1** (rejected — never counted as a pass); incomplete_no_def **18**; assertion_failure **7**.
- Dominant failure: **incomplete_no_def**; tree_serialize 1/1 (preserved True).

## Decision

| Gate | Status |
|---|---|
| score_improves_over_5of32 | FAIL |
| tool_call_rate_retained | PASS |
| no_tool_call_not_dominant | PASS |
| fake_pass_rejected | PASS |
| tree_serialize_preserved | PASS |
| artifact_safety | PASS |

**HOLD/REJECT** — gate(s) not satisfied: score_improves_over_5of32. Diagnostic: strict-verified 32-task stays at 5/32 — the dominant body failure is `incomplete_no_def` (incomplete_no_def=18, assertion_failure=7, fake_pass=1 rejected). Tool-call emission is recovered; the residual bottleneck is generating a correct implementation BODY, which inference-time control cannot manufacture (that requires generation/training, not a controller).

See `classification.csv`, `claim_boundary.md`.
