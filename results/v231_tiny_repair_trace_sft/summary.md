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

## Phase 3 — Evaluation

- **NOT RUN** — GPU-gated (`scripts/eval_v231_repair_sft.py`). Planned: (1) v2.30 repair validation slice; (2) frozen 32-task benchmark; (3) hard tree subset; (4) tree_serialize 3/3 format-control check. Comparisons: base+verifier vs adapter+verifier vs adapter without verifier. No evaluation metrics are fabricated.

## Decision

**HOLD** — dataset export + GPU-gated trainer/eval harness are delivered and validated offline (export runs; trainer/eval skip cleanly with no fabricated metrics), but the actual pilot run is deferred to a GPU host. Promotion requires a stable training run + non-regressing 32-task benchmark, which this CPU-only environment cannot produce.

See `dataset.csv`, `claim_boundary.md`.
