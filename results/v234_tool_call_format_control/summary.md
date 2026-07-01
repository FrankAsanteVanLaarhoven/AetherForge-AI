# v2.34 — Tool-Call Format Control

Isolates tool-call emission/control. v2.31, v2.32, and v2.33 all failed primarily through `no_tool_call` collapse on the frozen 32-task benchmark. v2.34 is NOT repair training — it is an inference-time, deterministic tool-call format controller applied to the v2.33 benchmark outputs, re-wrapping the model's already-emitted code into the `execute_code({...})` schema (it never invents code). Success = recover valid tool-call emission AND improve over v2.33's 5/32 — not model/repair improvement.

## Results (offline counterfactual over the v2.33 32-task transcripts)

| Metric | baseline (v2.33, no control) | controlled (v2.34) |
|---|---|---|
| 32-task pass | 5/32 | 5/32 (champion 23) |
| tool_call_rate | 0.188 | 0.969 |
| no_tool_call | 26 | 1 |
| execute_code_rate | — | 0.188 |
| invalid_tool_json | — | 6 |

- Recovered tool-calls: **25**; recovered passes: **0**; unsafe/ambiguous repairs rejected: **1**.
- no_tool_call dominant after control: **False**.
- Controlled failure reasons: {'wrapped_no_passing_solution': 25, 'no_tool_call': 1, 'invalid_tool_json': 1}.
- Hard-tree 1/3 → 1/3; tree_serialize preserved True (1/1).

## Decision

| Gate | Status |
|---|---|
| no_tool_call_not_dominant | PASS |
| tool_call_rate_improves | PASS |
| score_improves_over_5of32 | FAIL |
| tree_serialize_preserved | PASS |
| no_unsafe_fabrication | PASS |
| artifact_safety | PASS |

**HOLD/REJECT** — gate(s) not satisfied: score_improves_over_5of32. Diagnostic: the tool-call FORMAT/emission bottleneck is RESOLVED (no_tool_call 26→1, tool-call rate 0.188→0.969), but the 32-task score is unchanged because the recovered calls wrap asserts without a passing solution body — the bottleneck has shifted from tool-call emission to solution generation.

_v2.34 makes no claim of model improvement, repair improvement, SOTA, SWE-bench success, or production readiness._

See `metrics.csv`, `claim_boundary.md`.
