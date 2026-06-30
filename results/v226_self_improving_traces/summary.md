# v2.26 — tree_serialize Representation Attack + Trace Factory

The same 3 logical tree-serializations in 4 output representations (exact_string, token_list, nested_list, json), eval'd at 3B-bf16 + structured verifier (clean config). Holds the algorithm constant, varies only the output FORMAT. Held-out benchmark tasks are not touched.

## Part A — pass rate by representation

### 3B-bf16

| Representation | pass / total | rate |
|---|---|---|
| exact_string | 3/9 | 33% |
| token_list | 7/9 | 78% |
| nested_list | 4/9 | 44% |
| json | 2/9 | 22% |

- exact-string 33% vs structural mean 48% (gap +15%).
- failure kinds: algorithm/control 18, format/correctness 2.

## Verdict

**FORMAT_SENSITIVE** — Output FORMAT strongly modulates success (22%→78%, spread 56%) for the IDENTICAL algorithm — so these tasks are heavily format/control-bound — but NOT simply 'string hard, structural easy': best `token_list` 78%, worst `json` 22%, exact-string middling (33%). The held-out `tree_serialize` difficulty is format-related (nested/structured output is the real cost) rather than exact-string-specific.

## Part B — trace factory (source-only; traces local-only / gitignored)

- Traces recorded: **36** (full agentic trajectories, schema: plan/candidate/verifier_signal/repair/final + quality + contamination guard).
- Trace-quality rates: {'plan_present': 1.0, 'base_case_present': 1.0, 'combine_step_present': 0.972, 'minimal_test_present': 1.0, 'repair_used_verifier_signal': 1.0}.
- Contamination guard: 0 held-out name/function/prompt/solution overlap (by construction).
- By representation/status: {'exact_string/fail': 6, 'exact_string/pass': 3, 'json/fail': 7, 'json/pass': 2, 'nested_list/fail': 5, 'nested_list/pass': 4, 'token_list/fail': 2, 'token_list/pass': 7}.

## Promotion decision

**PROMOTE (representation finding)** — output format strongly modulates success for the identical algorithm; nested/structured output is the real cost (flat token-list easiest, json hardest). `tree_serialize` is substantially format/control-bound. Actionable: prefer the model's robust format, or train format-robustness across representations.
**Trace factory: PROMOTE** — traces are complete, contamination-guarded, verifier-labelled, and usable as future SFT/preference data.

See `by_representation.csv`, `claim_boundary.md`.
