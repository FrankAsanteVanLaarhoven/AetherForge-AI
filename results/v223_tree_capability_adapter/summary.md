# v2.23 — Targeted Tree Capability Adapter Summary

**Baseline:** 16.3/32. **Champion+verifier (v2.22):** hard tasks serialize 0/3, from_list 2/3, max_path_sum 1/3; full-32 mean 22.0. The adapter is a fresh LoRA on the MERGED champion (no LoRA-on-LoRA; champion not modified), evaluated under the same v2.22 verifier mode + v2.19c retrieval.

Training data: 49 contamination-guarded same-family-different-task tree repair traces (0 overlap with the benchmark). The hard tasks stayed evaluation-only.

## Hard tree tasks (the decision metric)

| Task | Baseline | Champion+verifier | Adapter+verifier | Adapter NO-verifier | Converted |
|---|---|---|---|---|---|
| v210_tree_serialize | 0/3 | 1/3 | 1/3 | 1/3 | no |
| v210_tree_from_list | 0/3 | 2/3 | 1/3 | 0/3 | no |
| v210_tree_max_path_sum | 0/3 | 1/3 | 0/3 | 0/3 | no |

**Hard-tree conversions (adapter+verifier): 0** (none).
Conversions that PERSIST without the verifier (weight-level): 0 (none).

## Full-32 (regression gate — not the promotion basis)

- Adapter+verifier full-32: [22, 23, 23] → mean 22.7/32 (std 0.47; v2.22 champion+verifier 22.0; baseline 16.3).
- Hard regressions (stable_pass → 0/3): none.
- Within v2.22 band (≥20.0): yes.

## Verdict

**NO_CONVERSION** — The targeted adapter did not overcome the residual capability-bound tree tasks under the contamination-guarded protocol.

### Notable observation

- The controlled adapter neither converted a hard task NOR regressed the benchmark (full-32 22.7 vs champion 22.0, no hard regressions). This contrasts with the project's prior retrains, which all regressed (v2.5 53.6%, v2.6 57.1%, Option A 64.3%): a small, low-LR, separate-adapter-on-merged-champion pilot is SAFE but, here, insufficient to crack these 3 tasks.
- Differences vs champion on the hard tasks (from_list 2/3→1/3, max_path 1/3→0/3) are best-of-3 flip-level, not meaningful — none of the three was ever a stable pass.
- The 3 tasks remain capability-bound; they may need more targeted data/steps or are at the 1.5B capability ceiling. The v2.22 repair traces remain the diagnostic record.

See `comparison.csv`, `hardtree.csv`, `claim_boundary.md`.
