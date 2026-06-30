# v2.32 — Claim Boundary

## Claimed

- v2.32 builds a mixed repair + tool-use-preservation SFT substrate with a split-loss trainer and a gated evaluation, to test whether tool-use can be preserved while adapting on repair traces.

## Status

- Mixed dataset: DONE and committed (summary only; dataset local-only).
- Training/eval: DONE on a GPU host.
- Benchmark gate: DONE.

## Not claimed

- No SWE-bench success, no production reliability, no RL training, no general SOTA, no frontier-level agent performance, no broad model improvement.
- No champion adapter or memory index created or overwritten; preservation traces are correct scaffolds on newly-authored non-held-out tasks, not held-out solutions.
