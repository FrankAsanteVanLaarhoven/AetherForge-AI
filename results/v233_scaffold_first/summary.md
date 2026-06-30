# v2.33 — Scaffold-First Tool-Call Preservation

Isolates scaffold/tool-call preservation. v2.31 (repair-only) and v2.32 (repair + preservation) both improved local repair but COLLAPSED the frozen 32-task agent benchmark with `no_tool_call` dominant. v2.33 trains ONLY on correct execute_code scaffold trajectories (no repair) to test whether tool-use and the 32-task benchmark can be preserved first. Success is preservation without regression — NOT repair improvement. Dataset/adapter local-only; only this summary is committed.

## Phase 1 — Scaffold dataset (committed evidence)

- Total: **32** | train **25** / val **7**.
- Objectives: {'tool_use_preservation': 32} (repair examples = **0**, must be 0).
- Task families: {'list_format': 3, 'kv_format': 2, 'container_format': 3, 'string_format': 1, 'json_format': 1, 'arithmetic': 4, 'recursion': 2, 'sequence': 2, 'scan': 2, 'tree_serialize_repr': 12}.
- Contamination guard violations: **0** (rejections {}).

## Phase 2 — Scaffold-only training

- Base `Qwen/Qwen2.5-Coder-1.5B-Instruct` | scaffold-only | loss trend [8.061253356933594, 6.9368537902832035, 6.057427978515625, 6.019522476196289, 4.384450912475586, 3.6475990295410154, 3.386882019042969, 3.349059295654297, 2.5846675872802733, 2.3868051528930665, 2.713628578186035, 2.273435592651367].

## Phase 3 — Evaluation (scaffold-first)

- Tool-use preservation: 0/7 (rate 0.0; tool_call_rate 0.857).
- 32-task: champion 23 vs adapter 5; tool_call_rate 0.188; execute_code_rate 0.188; no_tool_call 26 (dominant True).
- Hard-tree 1/3; tree_serialize 3/3 preserved True.
- Failure reasons: {'no_tool_call': 26, 'invalid_tool_json': 1}.

## Decision

| Gate | Status |
|---|---|
| training | PASS |
| tool_use_preserved | FAIL |
| artifact_safety | PASS |
| benchmark_non_regression | FAIL |

**HOLD/REJECT** — gate(s) not satisfied: tool_use_preserved, benchmark_non_regression. no_tool_call is the dominant failure mode (the v2.31/v2.32 collapse persists). Scaffold preservation must hold before repair traces return.

_Repair validation is an optional diagnostic only and is NOT a v2.33 promotion gate._

See `scaffold.csv`, `claim_boundary.md`.
