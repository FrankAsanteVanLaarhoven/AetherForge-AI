# v2.31 — Tiny Repair-Trace SFT Pilot

First training milestone: a tiny supervised fine-tuning pilot on the clean v2.30 repair-trace substrate. Target behaviour: given task + failed candidate + structured verifier signal, produce repair plan + corrected solution. Bounded pilot — not SOTA, not production, not RL.

## Phase 1 — SFT dataset export (committed evidence)

- Total available records: **50**
- Training records: **40** | validation records: **10**
- Format-repair: **28** | algorithmic-repair: **22**
- Task families: 10 ({'tree_serialize_repr': 9, 'list_format': 6, 'kv_format': 4, 'container_format': 6, 'string_format': 2, 'json_format': 1, 'arithmetic': 8, 'recursion': 6, 'sequence': 2, 'scan': 6})
- Sources: {'v229': 9, 'v230': 41}
- Rejection reasons: {}
- Contamination guard violations: **0** (any overlap or held-out-name record is rejected at export).

## Phase 2 — Tiny SFT run

- **NOT RUN in this environment** — CPU-only torch (`+cpu`, no CUDA). The trainer (`scripts/train_v231_repair_sft.py`) is GPU-gated and skips cleanly here. Run on a GPU host to produce the adapter at `outputs/v231_tiny_repair_trace_sft/` (local-only). No training metrics are fabricated.

## Phase 3a — Repair validation

- **NOT RUN** — GPU-gated (`scripts/eval_v231_repair_sft.py`).

## Phase 3b — Delegated benchmark gates (frozen 32-task / hard tree / tree_serialize)

- **NOT RUN** — required for full promotion. Run on the GPU host (adapter must exist):

  ```bash
  python scripts/eval_v231_repair_sft.py --benchmarks \
    --base <base> --adapter outputs/v231_tiny_repair_trace_sft/adapter
  ```
  (delegates to evaluate_code_agent.py: 32-task `data/v210_clean_repair_generalisation_tasks.jsonl`,
  hard-tree subset, and the v2.26 representation tasks for tree_serialize 3/3.)

## Decision

| Gate | Status |
|---|---|
| Training stable | PENDING |
| Repair validation (adapter ≥ base) | PENDING |
| Artifact safety (contamination 0) | PASS |
| 32-task non-regression + tree_serialize 3/3 | PENDING |

**HOLD** — pilot not run in this environment (CPU-only, no CUDA). Dataset export is committed; the GPU-gated trainer/eval/benchmark harness is ready. No metrics are fabricated.

See `dataset.csv`, `claim_boundary.md`.
