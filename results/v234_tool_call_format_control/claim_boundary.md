# v2.34 — Claim Boundary

## Claimed

- v2.34 provides a deterministic, inference-time tool-call format controller and an offline measurement of how much valid execute_code/tool-call emission it recovers on the v2.33 frozen-32-task transcripts. The controller only re-wraps code the model already emitted; it never invents code, so recovered passes are genuine (require a runnable solution).

## Not claimed

- No model improvement, no repair improvement, no SOTA, no SWE-bench success, no production readiness. v2.34 is a control/measurement experiment, not training.
- No model, champion adapter, or memory index was created or overwritten; no new generation was performed (the experiment reuses existing local benchmark transcripts).

## Finding

- The format/emission bottleneck is resolvable inference-time (no_tool_call no longer dominant, tool-call rate materially up), but the 32-task score is unchanged: the recovered calls wrap asserts without a passing solution, so the bottleneck shifts to solution generation. Decision: HOLD/REJECT per the strict gate.
