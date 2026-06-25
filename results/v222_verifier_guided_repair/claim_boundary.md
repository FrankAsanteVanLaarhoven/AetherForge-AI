# v2.22 — Claim Boundary

## Verdict

**NO_CONVERSION.** Bounded verifier-guided repair did not overcome the capability-bound tree failures; the repair traces localize the failure for a future targeted fine-tune.

## What this measures

- A precise VERIFIER signal (failing assert + expected/actual, exception+line, or
  no-output) plus a bounded repair budget with no-repeat enforcement, on top of the
  execution-plan prompt and the SAME v2.19c retrieval. No model weights, no new index.
- Decision metric: repair-attributable conversion of a capability-bound tree task
  (passing trajectory shows a VERIFIER FAIL before the PASS), not aggregate mean.
- The verifier compares the model's OWN code against its OWN diagnostic asserts — no
  reference solution, so the benchmark stays independent.

## Not claimed

- No SWE-bench success; no production reliability; no model-weight improvement.
- No frontier-model superiority; general tree reasoning is NOT solved.
- Aggregate-mean movement is not a promotion basis.
- Bounded to the 32-task benchmark, best-of-3.
- No AI/tool/vendor attribution.

## Next direction

The capability-bound tree tasks resist coverage (v2.19c), planning (v2.21/b), and
now bounded verifier-guided repair. The per-iteration repair traces (VERIFIER
signal + the model's attempts) are a precise, verified dataset of WHERE execution
breaks — the correct minimal input for a targeted LoRA fine-tune (v2.23), rather
than guessing.
