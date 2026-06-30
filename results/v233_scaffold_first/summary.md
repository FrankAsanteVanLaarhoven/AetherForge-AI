# v2.33 — Scaffold-First Tool-Call Preservation

Isolates scaffold/tool-call preservation. v2.31 (repair-only) and v2.32 (repair + preservation) both improved local repair but COLLAPSED the frozen 32-task agent benchmark with `no_tool_call` dominant. v2.33 trains ONLY on correct execute_code scaffold trajectories (no repair) to test whether tool-use and the 32-task benchmark can be preserved first. Success is preservation without regression — NOT repair improvement. Dataset/adapter local-only; only this summary is committed.

## Phase 1 — Scaffold dataset (committed evidence)

- Total: **32** | train **25** / val **7**.
- Objectives: {'tool_use_preservation': 32} (repair examples = **0**, must be 0).
- Task families: {'list_format': 3, 'kv_format': 2, 'container_format': 3, 'string_format': 1, 'json_format': 1, 'arithmetic': 4, 'recursion': 2, 'sequence': 2, 'scan': 2, 'tree_serialize_repr': 12}.
- Contamination guard violations: **0** (rejections {}).

## Phase 2 — Scaffold-only training

- **NOT RUN** — CPU-only; GPU-gated trainer skips cleanly (no fabricated metrics).

## Phase 3 — Evaluation (scaffold-first)

- Tool-use preservation: **NOT RUN** — GPU-gated.
- Benchmark gate: **NOT RUN** — required for PROMOTE: `python scripts/eval_v233_scaffold_sft.py --benchmarks --base <base> --adapter outputs/v233_scaffold_first_sft/adapter`.

## Decision

| Gate | Status |
|---|---|
| training | PENDING |
| tool_use_preserved | PENDING |
| artifact_safety | PASS |
| benchmark_non_regression | PENDING |

**HOLD** — not run in this environment (CPU-only, no CUDA). Scaffold dataset is committed; the GPU-gated trainer / eval / benchmark harness is ready. No fabricated metrics.

_Repair validation is an optional diagnostic only and is NOT a v2.33 promotion gate._

See `scaffold.csv`, `claim_boundary.md`.
