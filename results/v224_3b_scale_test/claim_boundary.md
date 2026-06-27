# v2.24 — Claim Boundary

## Verdict

**SCALE_HELPS.** The 3B BASE alone converts 1 hard task(s) the 1.5B champion could not — the residual capability wall is SCALE-DEPENDENT, not absolute. Targeted training then converts a further 1 task(s) (`v210_tree_max_path_sum`) on top of the 3B base — the SAME contamination-guarded recipe that did nothing at 1.5B (v2.23/v2.23b). Scale did not just solve a task; it made the model trainable for these patterns.

## What this measures

- Qwen2.5-Coder-3B-Instruct (base, and a fresh LoRA on it trained on the same v2.23b
  contamination-guarded same-family-different-task tree data) under the v2.22 verifier
  mode + v2.19c retrieval. Decision metric: stable (3/3) conversion of a hard task.
- The 1.5B champion is fine-tuned; the 3B is base — so a 3B-base win is a STRONG scale
  signal (3B wins despite no champion-style fine-tuning).

## Not claimed

- No SWE-bench success; no production reliability; no frontier superiority; no broad SOTA.
- 3B full-32 is not directly comparable to the 1.5B 16.3 baseline (different model).
- The frozen 1.5B champion (23/28 = 82.1%) is unchanged; the 3B artifacts are a separate
  scale probe, not a champion replacement.
- Bounded to the 32-task benchmark, best-of-3.
