# v2.23 — Claim Boundary

## Verdict

**NO_CONVERSION.** The targeted adapter did not overcome the residual capability-bound tree tasks under the contamination-guarded protocol.

## What this measures

- A small fresh LoRA (50 steps, lr 1e-5) on the MERGED champion, trained on 49
  contamination-guarded same-family-different-task tree repair traces. Champion not
  modified; protected indexes untouched.
- Decision metric: hard-tree conversion (stable 3/3) under a strict regression gate
  (no hard regression; full-32 not materially below the v2.22 band).
- Required ablation: adapter with vs without the structured verifier.

## Contamination guard

- scripts/check_v223_contamination.py asserts 0 overlap (function name, benchmark
  name leakage, prompt, hard-task tokens). The 3 hard tasks are evaluation-only.

## Not claimed

- No SWE-bench success; no production reliability; general tree reasoning NOT solved.
- No frontier-model superiority; no broad SOTA.
- The frozen champion (23/28 = 82.1%) is UNCHANGED; a promoted adapter is an additive
  artifact, not a champion replacement.
- Bounded to the 32-task benchmark, best-of-3.
