# v2.25 — 7B-Scale Capability Test Summary (QLoRA / 4-bit)

Extends the v2.24 scale curve to Qwen2.5-Coder-7B (4-bit NF4, fits 16GB). Same v2.22 verifier mode + v2.19c retrieval + v2.23b contamination-guarded data. The 1.5B champion and 3B artifacts are untouched.

## Hard tree tasks across the scale curve (the decision metric)

| Task | 1.5B champ | 3B base | 3B adapter | 7B base | 7B adapter | 7B adapter NO-ver |
|---|---|---|---|---|---|---|
| v210_tree_serialize | 1/3 | 1/3 | 0/3 | 2/3 | 2/3 | 1/3 |
| v210_tree_from_list | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| v210_tree_max_path_sum | 1/3 | 2/3 | 3/3 | 3/3 | 1/3 | 0/3 |

**7B conversions — base: 1 (`v210_tree_max_path_sum`); adapter: 1 (`v210_tree_from_list`).**

## Full-32 across the scale curve

| Scale | mean |
|---|---|
| 1.5B champion+verifier | 22.0 |
| 3B base+verifier | 26.3 |
| 3B adapter+verifier | 29.0 |
| 7B base+verifier | 21.3 [22, 18, 24] |
| 7B adapter+verifier | 31.0 [31, 31, 31] |

- 7B adapter regressions vs 7B base (3/3 → 0/3): none.

## Verdict

**SCALE_CONTINUES** — 7B converts 2 hard task(s) but `tree_serialize` remains the holdout even at 7B — the scale trend continues yet exact-string serialization resists this class.

### Key findings (honest)

- **The 7B QLoRA adapter reaches 31.0/32 ([31, 31, 31]) — the highest AND most stable aggregate in the whole arc** (vs 1.5B 22.0, 3B-adapter 29.0). It recovers decisively from the 7B-4bit BASE's unstable 21.3 ([22, 18, 24]): targeted training stabilised the 7B's agent-format following and compounded with scale.
- **4-BIT CONFOUND (key caveat).** 7B runs in 4-bit NF4 (the only fit on 16GB); 3B ran in bf16. The 7B-4bit BASE aggregate (21.3) is BELOW 3B-bf16 base (26.3) — 4-bit degradation offsets the parameter gain on the base. So the clean like-for-like scale win remains v2.24 (1.5B-bf16 → 3B-bf16: 22.0 → 26.3/29.0); v2.25 is a bounded, quantization-confounded extension. The adapter's 31.0 shows QLoRA recovers and exceeds even in 4-bit, but is not strictly comparable to the 3B-bf16 figures.
- **Hard tasks remain only partially / unstably solved at 7B-4bit, with a tradeoff.** The 7B base converts `tree_max_path_sum` (3/3); the adapter converts `tree_from_list` (3/3) but REGRESSES `tree_max_path_sum` to 1/3 — no single 7B-4bit config converts two of the three at once. `tree_serialize` (exact string format) never converts at any scale (best 2/3) — the deepest holdout.

### Scale trend

Across 1.5B → 3B → 7B the aggregate trends UP when targeted training is applied (22.0 → 29.0 → 31.0), and individual hard tasks fall at higher scale — but `tree_serialize` resists throughout, and the 4-bit constraint at 7B prevents a clean like-for-like base comparison on this hardware.

See `comparison.csv`, `hardtree.csv`, `claim_boundary.md`.
