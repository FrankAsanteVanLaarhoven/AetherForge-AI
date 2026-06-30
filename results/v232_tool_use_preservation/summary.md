# v2.32 — Tool-Use Preservation During Repair-Trace Adaptation

Mixes the genuine repair traces with tool-use / scaffold preservation traces (correct execute_code trajectories on non-held-out tasks) under a SPLIT LOSS (repair objective + tool-use preservation objective), to learn repair WITHOUT eroding the tool-call behaviour the frozen 32-task benchmark depends on. Not SOTA; bounded pilot. Dataset/adapter local-only; only this summary is committed.

## Phase 1 — Mixed dataset (committed evidence)

- Total: **82** | train **65** / val **17**.
- Objectives: repair **50** + tool-use preservation **32** (mix 0.61/0.39; preservation loss weight 1.0).
- Preservation families: {'list_format': 3, 'kv_format': 2, 'container_format': 3, 'string_format': 1, 'json_format': 1, 'arithmetic': 4, 'recursion': 2, 'sequence': 2, 'scan': 2, 'tree_serialize_repr': 12}.
- Validation split: repair 10 + preservation 7.
- Contamination guard violations: **0**.

## Phase 2 — Split-loss training

- **NOT RUN** — CPU-only environment; GPU-gated trainer skips cleanly (no fabricated metrics).

## Phase 3 — Evaluation

- **NOT RUN** — GPU-gated (`scripts/eval_v232_mixed_sft.py`).
- Benchmark gate: **NOT RUN** — required for PROMOTE: `python scripts/eval_v232_mixed_sft.py --benchmarks --base <base> --adapter outputs/v232_tool_use_preservation_sft/adapter`.

## Decision

| Gate | Status |
|---|---|
| training | PENDING |
| repair_improved | PENDING |
| tool_use_preserved | PENDING |
| artifact_safety | PASS |
| benchmark_non_regression | PENDING |

**HOLD** — not run in this environment (CPU-only, no CUDA). Mixed dataset is committed; the GPU-gated split-loss trainer / eval / benchmark harness is ready. No fabricated metrics.

See `mix.csv`, `claim_boundary.md`.
