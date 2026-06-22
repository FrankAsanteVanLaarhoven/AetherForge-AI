# AetherForge Evidence Closure Certificate

**Research arc:** v2.6 – v2.13
**Status:** CLOSED
**Date:** 2026-06-22

This document is an internal research audit record summarising what was tested, what
passed, what failed, what remains unproven, and what is concluded for the AetherForge
v2.6–v2.13 experimental arc.

---

## What Was Tested

| Experiment | Version | Question asked |
|---|---|---|
| Retraining audit | v2.5 / v2.6 | Does continued training with agent data improve the champion? |
| Champion preservation | v2.7 | Is memory load-bearing? Is merge safe? What is optimal k? |
| Hyperparameter / prompt audit | v2.8 | Can top-k tuning or prompt changes beat the champion? |
| Repair memory diagnostic | v2.9 | Do targeted repair records fix known failures? |
| Clean generalisation | v2.10 | Does the repair index generalise to unseen tasks? |
| Routing audit | v2.11 | Can routing strategies selectively apply repair memory? |
| Manuscript packet | v2.12 | Is the evidence packaged for reproducibility? |
| Paper draft | v2.13 | Is the evidence written up as a research paper? |

---

## What Passed

| Finding | Evidence | Claim type |
|---|---|---|
| Memory retrieval is load-bearing | 82.1% vs 64.3% (−17.8 pp without memory) | **Clean, supported** |
| merge_and_unload is safe | Delta ≤3.5 pp vs unmerged | **Clean, supported** |
| k=4 is optimal for this index | k=1 −10.7 pp, k=5 −3.5 pp | **Clean, supported** |
| Repair memory fixes known failures | 27/28 = 96.4% diagnostically | **Diagnostic — not clean champion** |
| 3 tasks recover under oracle routing | 23/32 = 71.9% oracle ceiling | **Diagnostic — not deployable** |

---

## What Failed

| Finding | Evidence | Decision |
|---|---|---|
| All retraining experiments | 50–64.3% vs champion 82.1% | Rejected — retraining not viable |
| All hyperparameter variants (v2.8) | ≤78.6% vs 82.1% champion | Rejected — champion config is optimal |
| Global repair-index promotion | 56.2% vs 62.5% champion on 32 tasks | Rejected — repair fails to generalise |
| Family router | 59.4% vs 62.5% champion | Rejected — within noise, directionally negative |
| Confidence router | 62.5% tied with champion | Rejected — no gain, too many false positives |

---

## What Remains Unproven

| Claim | Status | Reason |
|---|---|---|
| Dense retrieval (CodeBERT/UniXcoder) improves routing | Not evaluated | Future work |
| Results hold on SWE-bench | Not evaluated | No multi-file patch evaluation completed |
| Results hold on larger benchmarks (200+ tasks) | Not evaluated | Current benchmarks too small for strong statistics |
| Champion generalises to other model families | Not evaluated | Only tested on Qwen2.5-Coder-1.5B |
| Memory lift at k>4 or with better embedder | Not evaluated | Embedder not replaced in this arc |

---

## What Is Concluded

The following conclusions are supported by the evidence and can be stated honestly:

> AetherForge's clean champion reaches **23/28 = 82.1%** on the frozen benchmark.
> A diagnostic repair index reaches 27/28 = 96.4%, but that result targets known
> failures and is not a clean held-out champion. On a 32-task clean generalisation
> benchmark, the original champion index remains stronger than the repair index.
> The final conclusion is that **verified memory retrieval is load-bearing**, while
> naïve retraining, global repair-memory promotion, and TF-IDF routing are
> insufficient for robust generalisation.

The following conclusions are NOT supported and must not be claimed:

- That 27/28 = 96.4% is a clean champion
- That any routing strategy beats the champion cleanly
- That results generalise to SWE-bench or production code
- That performance scales to larger models or different architectures

---

## Benchmark Integrity Finding

One frozen benchmark task (`tree_depth_tuple`) contains a spec-conflicted assertion:
the prompt expects depth 3, but by the stated recursive rule the correct value is 4.
This was discovered during v2.9, reported in `docs/v2.9_memory_repair_split.md`, and
documented in `results/v212_manuscript_packet/03_failure_analysis.md`. It does not
affect the overall champion score claim (FAIL on this task is conservative; the
corrected audit score is 24/28 = 85.7%, not lower than 23/28).

---

## Recommended Future Work

1. **Dense code retrieval** — Replace TF-IDF with CodeBERT, UniXcoder, or
   nomic-embed-code. This is the most direct path to addressing retrieval noise.

2. **Operation-aware memory metadata** — Tag records with algorithm family labels.
   Exact-match family routing would correctly gate at least 2 of the 3 oracle-recoverable
   tasks without TF-IDF false positives.

3. **SWE-bench infrastructure** — Extend the evaluation harness to multi-file repository
   patch generation for real-world relevance.

4. **Larger clean benchmark** — 200+ tasks to reduce sampling variance below ±1 task
   and enable reliable statistical effect-size estimates.

---

## Certification

This certificate records that the AetherForge v2.6–v2.13 research arc has been closed
with a complete evidence trail. All positive, negative, and diagnostic results are
documented with root causes. No overclaims are made. The evidence package is
reproducible via `results/v212_manuscript_packet/05_reproducibility_commands.md`.

**Author:** Frank Asante Van Laarhoven
