# v2.23b — Claim Boundary

## Verdict

**CEILING.** Scaling the targeted training (~2x data, 3x steps) still did not convert any hard tree task. Combined with the v2.23 pilot, this indicates the residual failures are at or near the 1.5B capability ceiling for these patterns under the contamination-guarded protocol, not simply data-limited.

## What this measures

- A scaled fresh LoRA (150 steps) on the merged champion, trained on 94 contamination-
  guarded same-family-different-task tree traces (27 distinct tasks). Champion untouched.
- Decision metric: hard-tree conversion (stable 3/3) under a strict regression gate.
- Directly extends the v2.23 pilot to test data-limited vs capability ceiling.

## Not claimed

- That these tasks are unsolvable at any scale/model — only that ~2x data + 3x steps under
  this protocol did not move them.
- No SWE-bench success; no production reliability; no frontier superiority; no broad SOTA.
- The frozen champion (23/28 = 82.1%) is unchanged; any adapter is additive, not a
  replacement.
- Bounded to the 32-task benchmark, best-of-3.
