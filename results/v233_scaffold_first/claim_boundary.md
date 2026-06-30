# v2.33 — Claim Boundary

## Claimed

- v2.33 builds a scaffold-only (no-repair) tool-call preservation substrate with a GPU-gated trainer and a gated evaluation, to test whether execute_code / tool-use behaviour and the frozen 32-task benchmark can be preserved before repair adaptation is reintroduced.

## Status

- Scaffold dataset: DONE and committed (summary only; dataset local-only).
- Training/eval/benchmark: DEFERRED — CPU-only; GPU-gated scripts skip cleanly. No metrics fabricated.

## Not claimed

- No repair improvement claim (repair objective is intentionally absent in v2.33).
- No SWE-bench success, production reliability, RL training, general SOTA, or frontier-level agent performance.
- No champion adapter or memory index created or overwritten; scaffolds are correct tool-call trajectories on newly-authored non-held-out tasks, not held-out solutions.
