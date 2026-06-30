# v2.25 — Claim Boundary

## Verdict

**SCALE_CONTINUES.** 7B converts 2 hard task(s) but `tree_serialize` remains the holdout even at 7B — the scale trend continues yet exact-string serialization resists this class.

## What this measures

- Qwen2.5-Coder-7B-Instruct in 4-bit NF4 (base, and a QLoRA adapter on it trained on the
  same v2.23b contamination-guarded same-family-different-task tree data), under the v2.22
  verifier mode + v2.19c retrieval. Decision metric: stable (3/3) hard-task conversion.
- Forms a 1.5B → 3B → 7B scale curve on the three residual tasks.

## Not claimed

- No SWE-bench success; no production reliability; no frontier superiority; no broad SOTA.
- 4-bit and bf16 numbers are not identical; 7B here is 4-bit (the feasible config on 16GB).
- Full-32 across scales uses the same inference stack but different base models — a scale
  curve, not a single-model ablation.
- The frozen 1.5B champion (23/28 = 82.1%) is unchanged; all scale artifacts are separate.
- Bounded to the 32-task benchmark, best-of-3.
