# v2.28 — Claim Boundary

## Claimed

- v2.28 establishes a trace-dataset SUBSTRATE for later self-improving training: a canonical, contamination-guarded, quality-scored record schema and the source-only tooling to generate it locally from the v2.26/v2.27 traces.

## Not claimed

- No model improvement, no SWE-bench success, no production reliability, no RL training, no general SOTA, no frontier-level agent performance.
- No model was trained; no weights, champion adapters, or memory indexes were touched.
- The generated dataset is local-only; only this curated summary is committed.

## Contamination

- Computed guard violations across the accepted+rejected set = 0 (name/function/prompt/solution/test overlap vs the 32-task benchmark and v2.26 slice).
