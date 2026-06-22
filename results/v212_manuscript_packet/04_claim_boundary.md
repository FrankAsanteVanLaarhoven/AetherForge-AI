# AetherForge — Claim Boundary

## What the evidence supports

### Claim 1 — Verified memory retrieval is load-bearing
**Evidence:** v2.7. Adapter-only = 64.3%, adapter + memory = 82.1%. Delta = +17.8 pp.
6 tasks pass only with memory; 2 are hurt by memory (retrieval noise).
**Claim:** "Offline verified memory retrieval adds +17.8 pp on the frozen 28-task benchmark,
concentrated in hard algorithm tasks."

### Claim 2 — LoRA merge is safe
**Evidence:** v2.7. Unmerged adapter = 78.6%, merged adapter = 82.1%. Difference within noise.
**Claim:** "merge_and_unload does not damage adapter performance on this benchmark."

### Claim 3 — Retraining with higher LR or more data hurts generalisation
**Evidence:** v2.5, v2.6. Five independent runs across 0–25% trace ratios. Best result 57.1%
(0% traces). Training loss did not distinguish runs (0.129–0.132).
**Claim:** "All merge-and-retrain experiments caused regressions. The original adapter's
generalisation depends on its training trajectory and is not recoverable through retraining."

### Claim 4 — Targeted repair memory fixes known failures diagnostically
**Evidence:** v2.9. 4/4 repair targets fixed. 27/28 = 96.4% with repair index on frozen benchmark.
**Claim:** "Verified repair examples, when retrieved at k=4, allow the model to produce
correct implementations for previously failing algorithmic patterns."
**Caveat:** This is a diagnostic result. The frozen benchmark is no longer independent for
this configuration because repair records target known failures.

### Claim 5 — Repair memory does not generalise at scale
**Evidence:** v2.10. Repair index 18/32 = 56.2% vs champion 20/32 = 62.5% on 32 clean tasks.
Interval_merge improves (+16.7 pp) but nested_dict, tuple_tree, rle regress.
**Claim:** "Globally adding repair records to the champion index causes net regressions on
unseen tasks because repair vocabulary contaminates retrieval for adjacent task families."

### Claim 6 — TF-IDF routing cannot selectively apply repair memory
**Evidence:** v2.11. Confidence router achieves 20/32 = 62.5% (tied with champion).
Oracle ceiling 23/32 = 71.9%. Confidence signal fires on family-level vocabulary,
not task-specific repair relevance.
**Claim:** "TF-IDF similarity margin is not a reliable signal for repair-index routing.
The confidence router's gains (+3 tasks) exactly cancel its losses (−3 tasks)."

## What the evidence does NOT support

### Not supported — SWE-bench performance
The system has not been evaluated on SWE-bench or any real-world repository patch task.
The frozen 28-task benchmark uses synthetic held-out tasks, not production bug fixes.

### Not supported — Production-grade reliability
Best-of-3 sampling on 28 tasks gives approximately ±2–3 task variance per run.
82.1% on 28 tasks = 23 correct. Individual task outcomes are probabilistic, not guaranteed.

### Not supported — General superiority
The system has not been compared to other models (GPT-4, Claude, Gemini) on the same
benchmark. Comparison claims require co-evaluation on the same tasks.

### Not supported — The 96.4% diagnostic as a clean held-out champion
The 27/28 = 96.4% result uses a repair index that targets the exact 4 failing tasks.
The frozen benchmark is not independent for this configuration.
The clean champion remains 23/28 = 82.1% with `memory/index_adapted`.

### Not supported — Generalisation beyond this benchmark distribution
All 28 frozen tasks and 32 clean tasks are hard single-function Python code tasks with
specific verify-pass protocols. Performance on other task types (multi-file, full-repo,
natural language QA) has not been measured.

## Precedence order for conflicting results

When two results appear to conflict, this precedence order applies:

1. **Clean held-out results** outrank diagnostic results.
   (23/28 outranks 27/28 as the performance claim.)

2. **Scale results** outrank small-N signals.
   (32-task v2.10 outranks 5-task v2.9 clean set.)

3. **Conservative estimates** outrank optimistic ones.
   (Champion 62.5% on 32 tasks is the baseline; repair 56.2% is the alternative.)

4. **Replicated results** outrank single-run results.
   (Outcomes consistent across v2.10 and v2.11 sub-evals are more reliable than
   single-run outcomes that flip between runs.)

## Version snapshot

| Component | Value |
|---|---|
| Model | `outputs/qwen15b_v27_champion_merged` |
| Index | `memory/index_adapted` (99 records) |
| k | 4 |
| Frozen benchmark | 23/28 = 82.1% |
| Clean 32-task benchmark | 20/32 = 62.5% |
| Oracle routing ceiling | 23/32 = 71.9% |
| Evaluation date | 2026-06-22 |
