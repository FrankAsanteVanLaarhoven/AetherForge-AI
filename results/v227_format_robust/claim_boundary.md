# v2.27 — Claim Boundary

## Verdict

**FORMAT_CONTROL_RESOLVES.**
A deterministic canonical-IR + structured-format-verifier layer converts `tree_serialize`
stably across all output formats and diagnoses/repairs all 7 injected fault classes.

## What this is / is not

- This is an INFERENCE-TIME format-control result, not a training result. No SFT, no
  preference optimization, no adapter, no model-weight change. The frozen 1.5B champion
  (23/28 = 82.1%) and all memory indexes are untouched ⇒ no regression is possible.
- The model baseline is mined from existing v2.26 transcripts; NO new model run.
- The canonical renderers are correct by construction; the measured claim is that FORMAT
  CONTROL (not algorithm) is the bottleneck — supported by 15/36 envelope-format failures
  (algorithm correct, agent scored fail) and the model's 0 successful genuine repairs.
- Bounded to the tree_serialize family and these formats. No SWE-bench, production, or
  frontier claim. Generated traces are local-only; nothing generated is committed.

## Contamination

- Computed guard over name/function/prompt/solution/test overlap vs the 32-task benchmark
  and the v2.26 slice: violations = 0.
