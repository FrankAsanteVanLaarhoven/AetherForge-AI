# v2.31 — Claim Boundary

## Claimed

- v2.31 provides an initial supervised repair-trace adaptation pilot: a contamination-clean SFT export (40 train / 10 val from 50 genuine repairs) plus a GPU-gated tiny-LoRA trainer and evaluation harness to test whether verifier-labelled repair traces can train local repair behaviour.

## Status

- Dataset export: DONE and committed (summary only; dataset local-only).
- Training + evaluation: DEFERRED — this environment is CPU-only (no CUDA); the trainer and eval scripts skip cleanly and fabricate no metrics. Run on a GPU host to complete the pilot.

## Not claimed

- No SWE-bench success, no production reliability, no RL training, no general SOTA, no frontier-level agent performance, no broad model improvement.
- No model weights, champion adapters, or memory indexes were created or overwritten.
- The traces are controlled perturbations of non-held-out functions; the pilot tests repair supervision, not new capability.
