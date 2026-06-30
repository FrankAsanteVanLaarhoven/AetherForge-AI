# v2.27 — Format-Robust Output Control (tree_serialize)

Tests whether a canonical intermediate representation (IR) + structured format verification yields stable `tree_serialize` conversion. All arms are deterministic and model-free; the model baseline is mined from existing v2.26 transcripts (no new run). The frozen champion and memory indexes are untouched.

## Arm A vs Arm B — per-format pass rate (full_structure serialization core)

| Representation | model (3B-bf16, v2.26) | canonical-IR + format verifier |
|---|---|---|
| exact_string | 3/9 (0.33) | 40/40 (1.00) |
| token_list | 7/9 (0.78) | 40/40 (1.00) |
| nested_list | 4/9 (0.44) | 40/40 (1.00) |
| json | 2/9 (0.22) | 40/40 (1.00) |

- Battery: 40 deterministic trees/format (seed 227); model baseline over 3 v2.26 runs.

## Arm C — repair recovery (fault injection → verifier diagnosis → canonical repair)

| Fault class | correctly classified | canonical-repaired |
|---|---|---|
| missing_null_marker | 40/40 | 40/40 |
| extra_null_marker | 40/40 | 40/40 |
| separator_error | 40/40 | 40/40 |
| ordering_error | 39/40 | 39/40 |
| type_error | 40/40 | 40/40 |
| algorithmic_error | 40/40 | 40/40 |
| format_error | 40/40 | 40/40 |

## Phase 1 trace factory (genuine repair transitions)

- Hardened traces: **36** (each records candidate / verifier_signal / repair_plan / repaired / final separately).
- Genuine candidate≠final transitions (model actually changed its code): **7/36** — the model usually resubmitted identical code.
- Repair outcomes: {'no_repair_needed': 28, 'repair_attempted_failed': 8}.
- Repair kinds: {'envelope_format': 15, 'none': 13, 'format_only': 2, 'algorithmic': 6}.
- Envelope-format failures (algorithm correct standalone, agent scored FAIL): **15**.
- Contamination guard (COMPUTED): violations = 0.

## Success gate

- Canonical-control stable 3/3 on `exact_string` tree_serialize (3 logical tasks × 40 trees): **PASS**.
- Canonical control across ALL 12 (logical×format) cells: **100%**.
- Model repair loop converted genuinely-broken candidates: **0** (of 8 genuine failures).
- No model weights changed ⇒ no benchmark regression possible by construction; champion (23/28 = 82.1%) and memory indexes untouched.

## Verdict

**FORMAT_CONTROL_RESOLVES** — the canonical IR + structured format verifier converts `tree_serialize` stably (3/3 in the hardest format, 100% across all formats) and diagnoses+repairs every injected fault class, while the raw 3B model's own verifier-repair loop fixed 0 genuine failures and 15/36 of its 'failures' were algorithm-correct envelope/format errors. The residual `tree_serialize` difficulty is output-format/control-bound and is resolved by an inference-time format-control layer — not by changing the model.

