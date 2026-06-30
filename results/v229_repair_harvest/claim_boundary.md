# v2.29 — Claim Boundary

## Claimed

- v2.29 produces genuine verifier-labelled repair traces (candidate fails → structured format verifier signal → canonical repair → verified pass) suitable for FUTURE format-repair or verifier-format training.

## Not claimed

- No model improvement, no SWE-bench success, no production reliability, no RL training, no general SOTA, no frontier-level agent performance.
- No model trained; no weights, champion adapters, or memory indexes touched.
- Failures are controlled output-FORMAT perturbations of known-correct serializers; the underlying algorithm is preserved. The traces demonstrate repair, not new capability.
- Generated traces/dataset are local-only; only curated summaries are committed.

## Contamination

- Harvest guard violations = 0; held-out evaluation tasks and solutions are never used as harvest inputs.
