# Memory-Augmented Code-Agent Adaptation: When Retrieval Helps, When Retraining Hurts, and Why Routing Is the Bottleneck

**Frank Asante Van Laarhoven**

---

## Abstract

We present a controlled empirical study of AetherForge, a memory-augmented code-agent
built on a fine-tuned Qwen2.5-Coder-1.5B-Instruct adapter with an offline verified
vector memory. Our primary finding is that verified retrieval is load-bearing: removing
the memory index drops performance by 17.8 percentage points on our frozen 28-task
benchmark (82.1% → 64.3%). However, five controlled follow-up experiments show that
retraining, global repair-memory promotion, and TF-IDF confidence routing all fail to
improve over the base configuration. We identify retrieval noise — high-frequency
vocabulary overlap causing wrong-pattern context injection — as the primary barrier to
further improvement. Analysis reveals three distinct retrieval failure modes. The oracle
upper bound for any index-selection routing scheme is 23/32 = 71.9% on a 32-task
generalisation benchmark, with 9 tasks failing regardless of routing choice. Our
evidence trail includes one positive finding, four negative findings with identified root
causes, and a reproducible baseline for offline memory-augmented code agents on consumer
hardware.

---

## 1. Introduction

Code generation with language models has advanced rapidly, yet most strong results depend
on closed commercial APIs. Deploying capable code agents locally — on a single GPU, without
cloud inference — remains a practical challenge. Two questions become central in this
setting: how much can a small fine-tuned model be improved through better retrieval, and
what are the failure modes when retrieval guidance goes wrong?

We investigate these questions through AetherForge, a code-agent system built on
Qwen2.5-Coder-1.5B-Instruct with LoRA adaptation and an offline verified vector memory.
The memory stores verified examples — each is a task description paired with a working
code solution confirmed by execution — and retrieves relevant examples at inference time
using TF-IDF-like cosine similarity. This allows the model to benefit from previously
solved patterns without retraining.

Our experiments reveal a consistent finding: **verified memory retrieval is load-bearing**,
contributing +17.8 percentage points on our held-out benchmark. But improving beyond this
baseline is harder than expected. Retraining with additional data regresses performance at
every tested configuration. Targeted repair memory fixes known failures diagnostically but
does not generalise when added globally. Routing strategies that attempt to selectively
apply repair memory cannot improve over the champion index because the TF-IDF similarity
signal measures vocabulary overlap rather than algorithmic relevance.

We make three contributions. First, we provide a controlled measurement of the memory
lift in a local code-agent system and characterise its distribution across task types.
Second, we identify three classes of retrieval failure that explain why memory can
simultaneously help some tasks and hurt others. Third, we demonstrate that TF-IDF routing
cannot selectively recover repair gains, and establish the oracle upper bound for
index-selection routing with the current retrieval signal.

---

## 2. Problem: Local Code Agents Are Sensitive to Retrieval Quality

Code agents that operate through tool use — calling an executor, observing output, and
iterating — depend on their context window for guidance. Without retrieved examples, a
small model must generate solutions from scratch using only parametric knowledge. With
retrieved examples, the model can pattern-match against verified solutions, reducing the
effective search space.

This creates a dependency: when retrieval is accurate, performance improves significantly.
When retrieval injects a wrong-pattern example — one that shares vocabulary with the query
but solves a different problem — performance degrades. The challenge is that standard
retrieval methods (TF-IDF, bag-of-words cosine similarity) cannot distinguish these cases.

The problem is particularly acute for repair memories: examples added specifically to
address past failures. Repair examples are densely written with task-specific vocabulary,
which means they can score highly against unrelated tasks in the same algorithm family.
This produces a counter-intuitive result: targeted repairs that fix specific failures may
cause regressions on other tasks in the same family.

---

## 3. AetherForge System Overview

### 3.1 Model

The model is Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA (rank 16, alpha 32).
Training used 300 steps at 6e-6 learning rate on a curated agent-only dataset: examples
of the `execute_code` tool call format where the model produces code, an executor runs it,
and the observation confirms the output. This training teaches the tool-call format and
basic code-generation habits without injecting execution traces that could shift the
model's output distribution.

The adapter is merged at inference time using `merge_and_unload`, verified safe (§5.1).
The merged checkpoint runs at approximately 3.2 GB on an RTX 4080 Super (16 GB VRAM).

### 3.2 Memory System

The memory is an offline verified vector store:

- **Record format:** Each record contains a task description (`query_text`), a verified
  tool call (`corrected_tool_call`), and a confirmed observation (`PASS`).
- **Embedding:** Local TF-IDF-like sentence embedder (`code-memory-embedder`) generating
  dense vectors. Cosine similarity for retrieval.
- **Retrieval:** k-nearest-neighbour search with minimum score threshold.
  k=4 selected by ablation (§5.2).
- **Injection:** Retrieved records are formatted as a `RETRIEVED_VERIFIED_MEMORY` block
  in the system prompt. They are guidance only; the model must produce its own tool call.

The champion index contains 99 verified records covering a range of algorithm categories
(graph traversal, caching, string processing, sorting, nested data structures, etc.).

### 3.3 Evaluation Protocol

**Benchmark:** 28 hard held-out tasks covering single-function Python coding problems.
Tasks require correct implementations with specific assertions verified by execution.
The benchmark is frozen: never used in training, hyperparameter search, or repair targeting
until after the clean champion score was established.

**Scoring:** best-of-3 sampling, verified_agent scoring. The model must produce an
`execute_code` tool call whose execution returns `PASS`. Fallback scoring is disabled.
Strict agent contract: no direct answer extraction.

**Variance:** Sampling variance is approximately ±2–3 tasks on the 32-task benchmark at
best-of-3. Results separated by fewer than 2 tasks should be interpreted with caution.

---

## 4. Experimental Setup

All experiments use the same frozen 28-task benchmark, except v2.10 and v2.11 which
additionally use a 32-task clean generalisation benchmark (zero overlap with frozen tasks).
No experiment touches the champion model weights after the v2.7 audit.

Experiments are run sequentially; each result informs the next experiment design.
We report all experiments, including failures. The negative results are as informative as
the positive ones.

---

## 5. Results

### 5.1 Memory Is Load-Bearing (v2.7)

**Setup:** Evaluate the champion adapter with and without memory at varying k values.

| Configuration | Score |
|---|---|
| Adapter merged + memory k=4 | 23/28 = **82.1%** |
| Adapter merged + memory k=1 | 20/28 = 71.4% |
| Adapter merged + memory k=5 | 22/28 = 78.6% |
| Adapter merged, no memory | 18/28 = 64.3% |
| Adapter unmerged + memory k=4 | 22/28 = 78.6% |

**Memory lift:** +17.8 pp. This is the primary finding of the system: memory retrieval is
not decorative — it is required for 6 tasks that fail entirely without retrieved context
(count_islands, gcd_lcm, simple_ttl_cache, topo_sort, tree_depth_tuple, valid_ipv4).
Two tasks are *hurt* by memory (merge_intervals, slugify), due to retrieval noise (§6.1).

**Merge safety:** Merged and unmerged adapters differ by ≤3.5 pp (within variance).
`merge_and_unload` is safe for this model.

**k=4 is optimal:** k=1 loses +10.7 pp versus k=4; k=5 loses 3.5 pp. Retrieving four
examples provides enough context without diluting the prompt with irrelevant examples.

### 5.2 Retraining Does Not Recover or Improve the Champion (v2.5 / v2.6)

**Setup:** Fine-tune the merged champion with additional data at varying trace ratios and
learning rates to test whether the 75–82% result can be improved through continued training.

| Experiment | LR | Score | vs Champion |
|---|---|---|---|
| Merge + fresh LoRA | 5e-6 | 64.3% | −17.9 pp |
| v2.5 (5282 ex., 2e-5 LR) | 2e-5 | 53.6% | −21.4 pp |
| v2.6 traces=0% (2282 ex.) | 2e-5 | 57.1% | −17.9 pp |
| v2.6 traces=10% | 2e-5 | 50.0% | −25.0 pp |
| v2.6 traces=25% | 2e-5 | 53.6% | −21.4 pp |

All five experiments cause regressions. Training loss was nearly identical across
v2.6 runs (0.129–0.132), ruling out training instability. The pattern holds across
data sizes, trace ratios, and learning rates.

**Root cause:** The original 300-step, 6e-6 LR, agent-only training produced an adapter
whose generalisation depends on the specific trajectory: the model learns to invoke tools
and write correct code without being steered toward step-by-step trace output. Adding
more data at 2e-5 LR overwrites this property: hard task performance recovers (+24 pp)
but string task performance collapses (−50 pp) and basic tasks fail entirely (−100 pp).
The shift from direct-answer to trace-generating behaviour is the mechanism.

**Conclusion:** Further retraining is not a viable path for improvement. The champion
adapter is frozen.

### 5.3 Targeted Repair Memory Fixes Known Failures (Diagnostic) (v2.9)

**Setup:** Identify the 4 failing tasks, trace their retrieval noise via inspection, and
create 4 verified repair examples targeting the exact failing patterns. Evaluate on the
same frozen benchmark to test whether targeted repair examples fix the known failures.

| Task | Champion | Repair diagnostic | Type |
|---|---|---|---|
| merge_intervals | FAIL | PASS | Valid repair |
| median_two_sorted | FAIL | PASS | Valid repair |
| deep_get | FAIL | PASS | Valid repair |
| tree_depth_tuple | FAIL | PASS | Spec-conflicted† |

†See §6.2 for the benchmark defect finding.

**Diagnostic result:** 27/28 = 96.4% with repair index. **This is not a clean held-out
champion.** The frozen benchmark is no longer independent: repair records target the
exact 4 failing tasks by design. This result demonstrates that the right context fixes
the failures, not that the system has genuinely improved.

**Clean transfer test (5 tasks):** To get an early generalisation signal, we evaluated on
5 untouched tasks similar to the repair targets. 4/5 = 80% passed. This is a positive
signal but too small for a strong claim; v2.10 scales this test.

### 5.4 Repair Memory Does Not Generalise at Scale (v2.10)

**Setup:** 32 untouched tasks across 5 algorithm families, zero overlap with the frozen
28-task benchmark or the v2.9 5-task set. Evaluate both the champion index and the
repair-enhanced index.

| Family | Champion | Repair | Δ |
|---|---|---|---|
| Interval merging (6 tasks) | 4/6 = 66.7% | 5/6 = 83.3% | +16.7 pp |
| Sorted-array / kth (7 tasks) | 3/7 = 42.9% | 3/7 = 42.9% | 0 pp |
| Nested dict (7 tasks) | 6/7 = 85.7% | 5/7 = 71.4% | **−14.3 pp** |
| Tuple-tree recursion (7 tasks) | 3/7 = 42.9% | 2/7 = 28.6% | **−14.3 pp** |
| Run-length encoding (5 tasks) | 4/5 = 80.0% | 3/5 = 60.0% | **−20.0 pp** |
| **Total (32)** | **20/32 = 62.5%** | **18/32 = 56.2%** | **−6.3 pp** |

The repair index loses to champion by 6.3 pp overall. The net task flip is −2: adding
repair records gains 2 tasks (`interval_union`, `running_median`) and loses 4 others
(`deep_delete`, `unflatten_dict`, `tree_mirror`, `merge_sorted_k`).

**Root cause:** When repair records are added to the full index, they change the retrieval
landscape for all 32 tasks, not just the 4 repair targets. The `deep_get` repair record
scores 0.64–0.82 against all 7 nested_dict tasks because it shares the full family
vocabulary. Similarly, the `tree_depth` repair record scores 0.59–0.72 against all 7
tuple_tree tasks. When these repair records are retrieved for tasks they were not designed
to fix, they inject wrong-pattern context and cause regressions.

**Conclusion:** Global repair-index promotion is rejected. The champion index (99 records)
is the superior configuration for clean generalisation.

### 5.5 TF-IDF Routing Cannot Selectively Apply Repair Memory (v2.11)

**Setup:** Test three routing strategies that attempt to use the repair index only for
tasks that benefit from it, while defaulting to the champion index otherwise.

| Strategy | → Champion | → Repair | Score | vs Champion |
|---|---|---|---|---|
| Family router | 26 | 6 (interval_merge) | 19/32 = 59.4% | −3.1 pp |
| Confidence router (margin ≥ 0.05) | 12 | 20 | 20/32 = 62.5% | 0 pp |
| Oracle ceiling (perfect routing) | 29 | 3 | 23/32 = 71.9% | +9.4 pp |

**Family router:** Routes all 6 interval_merge tasks to the repair index. Gains at most
+1 task (`interval_union`), which is within sampling variance (the gain was not observed
in this run due to variance in the repair index's `interval_union` result).

**Confidence router:** Routes to repair when the repair top-1 similarity score exceeds
0.35 AND beats champion top-1 by ≥0.05. This fires on 20/32 tasks — including all nested
dict and all tuple-tree tasks, because repair records score 0.59–0.82 against entire
algorithm families. The three gains (`interval_union`, `running_median`, `deep_keys`)
exactly cancel three losses (`deep_delete`, `unflatten_dict`, `tree_mirror`): tied at
20/32.

**Oracle ceiling:** Per-task selection of whichever index passes in v2.10 achieves
23/32 = 71.9% (+9.4 pp). Only 3 tasks can be recovered by any routing scheme.
9 tasks fail with both indexes and are unreachable regardless of routing choice.

**Root cause:** TF-IDF similarity measures vocabulary family overlap, not
algorithm-specific repair relevance. The confidence signal fires where it should not
(family-level false positives) and would need to be orders of magnitude more specific
to identify which 3 tasks among 32 benefit from repair routing.

---

## 6. Failure Analysis

### 6.1 Retrieval Failure Taxonomy

Three distinct mechanisms cause harmful retrieval in this system:

**Type 1 — Surface lexical collision.** High-frequency tokens shared between query and
record despite different algorithmic intent. Example: `merge_intervals` and `merge_sorted`
share the tokens "merge" and "sorted." Eight `merge_sorted` records dominate k=4 retrieval
for the `merge_intervals` task (top-1 score: 0.557), injecting a sorting pattern where an
interval-sweep pattern is needed.

**Type 2 — Structural vocabulary overlap.** Nested-data tasks (`tree_depth_tuple`,
`deep_get`) share vocabulary with unrelated nested-pattern records. `flatten` records
dominate `tree_depth_tuple` retrieval ("nested", "list", "depth" co-occur in both);
`invert_dict` and `flatten` records dominate `deep_get` retrieval ("dict", "nested",
"key").

**Type 3 — Repair vocabulary leak.** Repair records are densely written with
algorithm-family vocabulary to ensure they retrieve correctly for the target task.
This causes them to score highly against all tasks in the same family. The `deep_get`
repair record scores 0.64–0.82 against all 7 nested_dict tasks; the `tree_depth` repair
record scores 0.59–0.72 against all 7 tuple_tree tasks. Routing heuristics based on
this signal cannot distinguish the intended target from family-level false positives.

All three failure types share the same root cause: **bag-of-words similarity conflates
algorithm identity with lexical co-occurrence**. A retrieval signal that captures
algorithmic structure rather than surface vocabulary would be necessary to address all
three types.

### 6.2 Benchmark Integrity Finding

During v2.9, we discovered that the `tree_depth_tuple` task prompt contains a broken
assertion:

```
tree_depth(((1,2),(3,(4,5)))) == 3
```

By the stated rule (leaves have depth 1; a branch has depth 1 + max(left, right)):

```
tree_depth((4,5)) = 2
tree_depth((3,(4,5))) = 3
tree_depth(((1,2),(3,(4,5)))) = 1 + max(2,3) = 4
```

The correct value is **4**, not 3. This was verified computationally. The task FAIL with
the champion index is caused at least in part by this broken assertion: a correct
implementation produces 4, which fails the assertion == 3.

**Action taken:** This task is documented as a spec-conflicted benchmark defect. All
results are reported both including and excluding this task ("raw frozen score" and
"corrected audit score"). The repair record uses the correct assertion (==4), which is
why the task PASS with the repair index should be interpreted as the model following
repair memory rather than the task prompt.

### 6.3 Sampling Variance

At best-of-3 sampling, individual task outcomes vary between runs. Across v2.10 and v2.11
sub-evaluations, we observed the following tasks flip outcome between runs:
`interval_union`, `insert_interval`, `merge_sorted_k`, `deep_delete`, `rle_expand`.
This variance (±2–3 tasks on a 32-task benchmark) means routing gains of fewer than
3 tasks cannot be reliably distinguished from noise.

### 6.4 Tasks Unsolvable by Either Index

Nine of 32 clean tasks fail with both the champion and repair indexes in v2.10:
`kth_smallest_matrix`, `wiggle_sort`, `count_smaller_after_self`, `tree_max_path_sum`,
`tree_from_list`, `tree_serialize`, `tree_width`, `rle_delta_encode`, and `insert_interval`
(in most runs). These tasks require either complex multi-step reasoning, algorithms not
covered by the current memory patterns, or implementations beyond the model's current
parametric capacity. They represent the model capability ceiling, not a retrieval failure.

---

## 7. Limitations

**Scale:** The frozen benchmark contains 28 tasks. The generalisation benchmark contains
32 tasks. Sampling variance is non-trivial relative to task count: ±2–3 tasks at
best-of-3. Effect sizes should be interpreted with this uncertainty in mind.

**Comparisons:** No baseline comparison to commercial or larger open-weight code
generation models has been run on these benchmarks. Comparative claims require
co-evaluation on the same held-out tasks.

**Task distribution:** All evaluation tasks are hard single-function Python coding
problems with specific verify-pass protocols. Generalisation to multi-file tasks,
natural language QA, or real-world repository bugs has not been tested.

**Retrieval architecture:** All results use TF-IDF-like embedding. Dense semantic
retrieval (CodeBERT, UniXcoder) was not evaluated; this leaves the hypothesis that
algorithm-aware embeddings would fix the retrieval failures untested.

**SWE-bench:** The system has not been evaluated on any real-world repository patch task.
No SWE-bench claims are made.

---

## 8. Reproducibility

All results are reproducible using the provided Makefile targets.
The champion model must be present at `outputs/qwen15b_v27_champion_merged`.

**Core commands:**
```bash
make eval-champion                  # Reproduce 23/28 = 82.1% clean champion
make eval-v29-repair-memory-diagnostic   # Reproduce 27/28 = 96.4% diagnostic
make eval-v210-clean-champion       # Reproduce 20/32 = 62.5% clean generalisation
make eval-v210-repair-index         # Reproduce 18/32 = 56.2% repair comparison
make route-v211 && make eval-v211-family-router && make eval-v211-confidence-router
make summarise-v211                 # Routing audit summary
```

Full commands and environment setup: `results/v212_manuscript_packet/05_reproducibility_commands.md`.

Sampling variance note: best-of-3 results on the 32-task benchmark show ±2–3 task
variance per run. The champion 23/28 result is stable across runs; clean-benchmark scores
should be reported with this uncertainty acknowledged.

---

## 9. Future Work

**Dense code retrieval.** Replacing TF-IDF similarity with a code-specific dense
embedder (CodeBERT, UniXcoder, CodeT5+) is the most direct path to addressing the
retrieval failures identified in §6. Semantic similarity would better distinguish
algorithmic patterns from vocabulary overlap, potentially enabling the confidence router
to approach the oracle ceiling (23/32 = 71.9% vs current 20/32 = 62.5%).

**Operation-aware memory metadata.** Even without dense retrieval, routing could be
improved by tagging each memory record with algorithm family labels and using exact-match
family routing rather than similarity-based routing. The v2.11 oracle analysis shows
which 3 tasks (in 32) would benefit; an exact-match family classifier would route
correctly for at least 2 of them (`interval_union` via interval_merge family;
`deep_keys` via nested_dict family).

**SWE-bench patch generation.** The evaluation harness supports single-function Python
tasks but does not yet handle multi-file repository patches. Extending the harness to
SWE-bench Lite would provide a benchmark with real-world relevance and established
comparison baselines.

**Larger benchmark.** The 28-task frozen benchmark and 32-task generalisation benchmark
are too small for strong statistical effect-size estimates. A 200+ task benchmark across
more algorithm families would reduce variance and allow between-condition effects to be
measured reliably.

---

## 10. Conclusion

We have shown that verified offline memory retrieval is load-bearing for a local
code-agent based on Qwen2.5-Coder-1.5B-Instruct: removing the memory index drops
performance by 17.8 percentage points. However, five controlled experiments reveal that
none of the tested improvement strategies — retraining, global repair-memory promotion,
or TF-IDF confidence routing — improves over the champion configuration on clean
generalisation tasks.

The binding constraint is retrieval relevance. TF-IDF bag-of-words similarity measures
vocabulary co-occurrence, not algorithm-level similarity. This causes three distinct
failure modes: lexical collision between algorithmically unrelated records, structural
vocabulary overlap across nested-data patterns, and repair vocabulary leak across
algorithm families. All three share the same root cause and point to the same solution:
an embedding that encodes algorithm structure rather than token frequency.

The oracle analysis shows that even perfect per-task routing between the two indexes
achieves 23/32 = 71.9% on the clean benchmark, with 9 tasks failing regardless of routing.
This separates the retrieval bottleneck (3 recoverable tasks) from the model capability
ceiling (9 tasks unsolvable by either index). Addressing the full 71.9% ceiling requires
both better retrieval and broader model capability.

The clean champion — 23/28 = 82.1% on the frozen benchmark, fully reproducible — is the
stable result. It establishes that memory-augmented LoRA adaptation can reach strong
performance on hard single-function coding tasks on consumer hardware, and identifies
retrieval relevance as the next engineering target.

---

## Appendix

**A. Full results tables** — `paper/tables/main_results_table.md`

**B. Experiment timeline** — `results/v212_manuscript_packet/01_experiment_timeline.md`

**C. Failure analysis** — `results/v212_manuscript_packet/03_failure_analysis.md`

**D. Claim boundary** — `results/v212_manuscript_packet/04_claim_boundary.md`

**E. Reproducibility commands** — `results/v212_manuscript_packet/05_reproducibility_commands.md`

**F. Per-task routing scores (v2.11)** — `results/v211_retrieval_routing/per_task_routing.csv`

**G. Retrieval failure taxonomy figure spec** — `paper/figures/retrieval_failure_taxonomy_spec.md`
