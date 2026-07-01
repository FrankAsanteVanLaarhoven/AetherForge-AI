# v2.35 — Claim Boundary

## Claimed

- v2.35 provides a strict solution-body verifier (benchmark-owned assertions only; the model's print('PASS') is never trusted) and a measurement of solution-body correctness after v2.34 tool-call recovery. It rejects fake PASS, missing asserts, and incomplete bodies.

## Not claimed

- No model improvement, no repair improvement, no SOTA, no SWE-bench success, no production readiness. No new generation, no training, no fabricated or injected solutions.
- No model, champion adapter, or memory index was created or overwritten.

## Finding

- Strict-verified 32-task stays at 5/32: tool-call emission is recovered, fake PASS is rejected, but the dominant body failure is `incomplete_no_def` — most bodies lack a correct implementation. The bottleneck is genuine solution generation, which an inference-time controller cannot manufacture. Decision: HOLD/REJECT.
