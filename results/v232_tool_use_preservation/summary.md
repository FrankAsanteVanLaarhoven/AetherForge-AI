# v2.32 — Tool-Use Preservation During Repair-Trace Adaptation

Mixes the genuine repair traces with tool-use / scaffold preservation traces (correct execute_code trajectories on non-held-out tasks) under a SPLIT LOSS (repair objective + tool-use preservation objective), to learn repair WITHOUT eroding the tool-call behaviour the frozen 32-task benchmark depends on. Not SOTA; bounded pilot. Dataset/adapter local-only; only this summary is committed.

## Phase 1 — Mixed dataset (committed evidence)

- Total: **82** | train **65** / val **17**.
- Objectives: repair **50** + tool-use preservation **32** (mix 0.61/0.39; preservation loss weight 1.0).
- Preservation families: {'list_format': 3, 'kv_format': 2, 'container_format': 3, 'string_format': 1, 'json_format': 1, 'arithmetic': 4, 'recursion': 2, 'sequence': 2, 'scan': 2, 'tree_serialize_repr': 12}.
- Validation split: repair 10 + preservation 7.
- Contamination guard violations: **0**.

## Phase 2 — Split-loss training

- Base `Qwen/Qwen2.5-Coder-1.5B-Instruct` | max_steps 80 | loss trend [7.893016815185547, 7.870266723632812, 7.353997802734375, 5.8299201965332035, 6.040426635742188, 5.389043807983398, 4.167635345458985, 4.409267807006836, 3.8454971313476562, 3.5199024200439455, 2.5650157928466797, 3.11093807220459, 2.94683780670166, 2.3865228652954102, 2.6706329345703126, 2.8651355743408202] | mix {'repair': 50, 'preservation': 32}.

## Phase 3 — Evaluation

- Repair: base 0/10 → adapter 2/10.
- Tool-use preservation: adapter 0/7 (floor 80%).
- Benchmark: champion 23 vs adapter 0; tree_serialize 3/3 preserved False.

## Decision

| Gate | Status |
|---|---|
| training | PASS |
| repair_improved | PASS |
| tool_use_preserved | FAIL |
| artifact_safety | PASS |
| benchmark_non_regression | FAIL |

**HOLD** — gate(s) not satisfied: tool_use_preserved, benchmark_non_regression. Promotion requires all gates; no fabricated metrics.

See `mix.csv`, `claim_boundary.md`.
