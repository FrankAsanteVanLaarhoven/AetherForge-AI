# v2.31 — Claim Boundary

## Claimed

- v2.31 provides an initial supervised repair-trace adaptation pilot: a contamination-clean SFT export (40 train / 10 val from 50 genuine repairs) plus a GPU-gated tiny-LoRA trainer and evaluation harness to test whether verifier-labelled repair traces can train local repair behaviour.

## Status

- Dataset export: DONE and committed (summary only; dataset local-only).
- Training: DEFERRED — CPU-only environment; GPU-gated trainer skips cleanly.
- Repair validation: PENDING (GPU).
- Benchmark gate (32-task / hard-tree / tree_serialize 3/3): DONE.

## Not claimed

- No SWE-bench success, no production reliability, no RL training, no general SOTA, no frontier-level agent performance, no broad model improvement.
- No model weights, champion adapters, or memory indexes were created or overwritten.
- The traces are controlled perturbations of non-held-out functions; the pilot tests repair supervision, not new capability.
