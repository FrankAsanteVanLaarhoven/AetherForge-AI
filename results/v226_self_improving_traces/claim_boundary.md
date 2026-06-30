# v2.26 — Claim Boundary

## Verdict

**FORMAT_SENSITIVE.** Output FORMAT strongly modulates success (22%→78%, spread 56%) for the IDENTICAL algorithm — so these tasks are heavily format/control-bound — but NOT simply 'string hard, structural easy': best `token_list` 78%, worst `json` 22%, exact-string middling (33%). The held-out `tree_serialize` difficulty is format-related (nested/structured output is the real cost) rather than exact-string-specific.

## What this measures

- The SAME 3 tree-serialization algorithms in 4 output representations, at 3B-bf16 +
  structured verifier. A per-format pass-rate gap isolates output-format difficulty from
  traversal capability. 7B-4bit is reference-only (quantization-confounded).
- A source-only trace factory recording contamination-guarded, verifier-labelled agentic
  trajectories for future SFT / preference optimization (NOT RL itself).

## Not claimed

- No SWE-bench success; no production reliability; no frontier superiority; no broad SOTA.
- Not a claim that the held-out `tree_serialize` is now solved — only what its difficulty
  is attributable to (format vs capability), measured on matched same-family tasks.
- Generated traces are local-only; no model weights / outputs / generated JSONL committed.
- The frozen 1.5B champion (23/28 = 82.1%) is untouched; bounded to best-of-3.
