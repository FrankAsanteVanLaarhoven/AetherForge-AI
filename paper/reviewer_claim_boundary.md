# Reviewer Claim Boundary

This document states precisely what the AetherForge paper claims and does not claim.
It is intended for pre-submission review and as a reference when responding to reviewers.

---

## The One Main Claim

> Verified offline memory retrieval is load-bearing for a locally-fine-tuned small code-agent:
> removing the memory index costs 17.8 percentage points on a held-out benchmark.
> Five controlled follow-up experiments show that this baseline cannot be improved through
> retraining, global repair-memory promotion, or TF-IDF routing, because the binding constraint
> is retrieval relevance rather than memory quantity.

Everything else in the paper is either supporting evidence, a negative result, or a boundary
statement.

---

## Supported Claims (with evidence)

| Claim | Evidence | Strength |
|---|---|---|
| Memory retrieval is load-bearing: +17.8 pp | v2.7 ablation, 5 configurations | Strong |
| k=4 is optimal for this index | k=1, k=4, k=5 comparison | Moderate (single index) |
| LoRA merge is safe: ≤3.5 pp delta | v2.7 merge vs. unmerged | Moderate |
| Retraining regresses at all tested configs | 5 configs, 17.9–25.0 pp regression | Strong |
| Global repair index loses on clean 32-task benchmark | v2.10, −6.3 pp | Moderate (one seed) |
| No routing strategy beats champion | v2.11, 3 strategies tested | Moderate |
| Oracle ceiling is 23/32 = 71.9% | v2.11 oracle analysis | Moderate (diagnostic) |
| Three retrieval failure modes identified | v2.8/v2.9/v2.10 failure inspection | Descriptive |
| tree_depth_tuple contains spec-conflicted assertion | Computational verification | Strong |

---

## Diagnostic Results (must be clearly labelled, not presented as clean champions)

| Result | Why it is diagnostic | Correct label |
|---|---|---|
| 27/28 = 96.4% (repair index, frozen benchmark) | Repair records target known frozen-benchmark failures; benchmark non-independent | "Diagnostic only" |
| 23/32 = 71.9% (oracle routing) | Oracle selects winning index per task using v2.10 ground truth; not deployable | "Oracle ceiling (diagnostic)" |
| 24/27 = 88.9% (corrected audit, excl. spec-conflicted) | Excludes one task with a known benchmark defect | "Corrected audit score" |

---

## Claims That Must Not Be Made

| Forbidden claim | Reason |
|---|---|
| "The system achieves 96.4% on the benchmark" (without diagnostic label) | Benchmark is non-independent for this result |
| "Memory retrieval improves generalisation robustly" | On 32-task clean benchmark, repair index loses to champion |
| "The system is ready for production use" | 28-task benchmark, best-of-3, single GPU; no production evaluation |
| "Results generalise to SWE-bench or real repositories" | Multi-file patch generation not evaluated in this arc |
| "Better than [model X]" without co-evaluation | No comparative evaluation on the same tasks |
| "The approach scales to larger models" | Only tested on Qwen2.5-Coder-1.5B |
| "Dense retrieval would not help" | Dense retrieval not tested; this would be an unsupported negative claim |
| "The memory system is production-grade" | Not the claim; the claim is about load-bearing lift in a research setting |

---

## Boundary Cases (handle with care)

**"Memory helps"** — TRUE, but only when retrieval is accurate. The claim should be:
"Verified memory retrieval is load-bearing when retrieval returns relevant examples."
The qualifier is important because the failure analysis shows that wrong-pattern retrieval
hurts performance.

**"Retraining is harmful"** — TRUE for the configurations tested, but not a universal
claim. The mechanism (distribution shift from higher learning rates on agent-format data)
is specific to this setup. The claim should be: "Retraining with additional data at 2e-5
LR caused output distribution shift that hurt string-task generalisation across all five
tested configurations."

**"TF-IDF routing cannot work"** — The correct claim is: "TF-IDF confidence routing
cannot distinguish algorithm-specific repair relevance from vocabulary-family co-occurrence
in this index. A denser or operation-aware retrieval signal might achieve different
results." The claim is about this signal and this index, not about all routing approaches.

---

## Responding to Likely Reviewer Questions

**Q: Why is 27/28 = 96.4% not the main result?**  
A: Because the benchmark is no longer independent. The repair index contains records
specifically designed to fix the 4 tasks that fail with the champion index. Evaluating
on the same 28 tasks after adding targeted fixes is not a held-out evaluation. The
32-task clean benchmark (zero overlap, constructed after repair records) is the
appropriate generalisation test, and on that benchmark the champion index outperforms
the repair index (62.5% vs 56.2%).

**Q: Is the 82.1% result statistically significant?**  
A: The benchmark contains 28 tasks, and sampling variance at best-of-3 is ±2–3 tasks.
The 17.8 pp memory lift (23/28 vs 18/28 = a difference of 5 tasks) exceeds the 3-task
variance threshold and is observed consistently across multiple runs. However, we do not
report p-values — the benchmark is too small for meaningful parametric testing.

**Q: How does this compare to HumanEval/MBPP/other benchmarks?**  
A: The frozen 28-task benchmark was constructed for this project and uses a different
evaluation protocol (verified_agent, strict tool-call contract, assertion-based). No
cross-benchmark comparison was run. Any comparison would require re-evaluating other
systems on the same tasks under the same protocol.

**Q: Why not use a larger model?**  
A: The research question is specifically about whether offline verified retrieval is
load-bearing for a small locally-fine-tuned model. A larger model would change both the
base capability and the retrieval interaction in ways that make the comparison ambiguous.
Larger-model experiments are listed as future work.
