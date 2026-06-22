# Memory-Augmented Code-Agent Adaptation: When Retrieval Helps, When Retraining Hurts, and Why Routing Is the Bottleneck

**Frank Asante Van Laarhoven**

---

## Abstract

We investigate memory-augmented adaptation for a local code-agent based on
Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA. The central finding is that
verified offline memory retrieval is load-bearing: removing the 99-record index
drops performance by 17.8 percentage points on a frozen 28-task benchmark
(82.1% → 64.3%). Five follow-up experiments test whether this baseline can be
improved. Continued LoRA training regresses at every tested configuration, by
17.9–25.0 pp, due to distribution shift in output format. Targeted repair memory
fixes four known failures diagnostically (96.4% on the same frozen benchmark),
but this result is not a clean held-out champion: the benchmark is no longer
independent because repair records target known failures. On a 32-task clean
generalisation benchmark with zero overlap, the original champion index (62.5%)
outperforms the repair-enhanced index (56.2%). Three routing strategies — family
routing, TF-IDF confidence routing, and oracle analysis — confirm that no
deployable strategy improves over champion on clean tasks. The oracle ceiling is
71.9%, with 9 of 32 tasks failing regardless of index selection. We identify
three retrieval failure modes rooted in the same cause: TF-IDF similarity measures
vocabulary co-occurrence rather than algorithmic relevance. The clean champion
(23/28 = 82.1%) is fully reproducible on a single consumer GPU.

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
using TF-IDF cosine similarity. Memory provides contextual guidance at inference time
without changing model weights, but it helps only when retrieval returns algorithmically
relevant examples; wrong-pattern retrieval degrades performance just as reliably as
accurate retrieval improves it. Specifically, bag-of-words similarity conflates vocabulary
co-occurrence with algorithmic relevance, and this is the root cause of all five negative
results in this study.

Our experiments reveal a consistent picture. Verified memory retrieval is load-bearing,
contributing +17.8 percentage points on our held-out benchmark. Retraining with additional
data regresses performance at every tested configuration. Targeted repair memory fixes
known failures diagnostically but does not generalise when added globally. Routing
strategies that attempt to selectively apply repair memory cannot improve over the champion
index because the TF-IDF similarity signal fires at the vocabulary-family level, not the
algorithm-instance level.

**Contributions.** (1) A controlled measurement of the memory lift in a local code-agent
system and characterisation of which task types benefit. (2) A three-class taxonomy of
retrieval failure modes, each traceable to the same root cause: bag-of-words similarity
conflates vocabulary co-occurrence with algorithmic relevance. (3) An oracle ceiling
analysis establishing that at most 3 of 32 clean tasks are recoverable by any
index-selection routing scheme, separating the retrieval bottleneck from the model
capability ceiling.

---

## 2. Problem: Local Code Agents Are Sensitive to Retrieval Quality

Code agents that operate through tool use — calling an executor, observing output, and
iterating — depend on their context window for guidance. Without retrieved examples, a
small model must generate solutions from scratch using only parametric knowledge. With
retrieved examples, the model can pattern-match against verified solutions, reducing the
effective search space.

This creates a hard dependency on retrieval accuracy. A model that retrieves the
`merge_sorted` pattern when asked to solve `merge_intervals` produces code that merges
two sorted lists rather than sweeping overlapping intervals — a failure invisible to the
retriever, because both tasks share the tokens "merge" and "sorted." When retrieval is
accurate, performance improves significantly. When it injects a wrong-pattern example,
performance degrades.

The problem is particularly acute for repair memories: examples added specifically to
address past failures. Repair examples are densely written with task-specific vocabulary,
which means they score highly against unrelated tasks in the same algorithm family.
This produces a counter-intuitive result: targeted repairs that fix specific failures cause
regressions on other tasks in the same family, because the repair record becomes the
nearest neighbour for the entire family, not just the intended target.

---

## 3. AetherForge System Overview

### 3.1 Model

The model is Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA (rank 16, alpha 32).
Training used 300 steps at 6e-6 learning rate on a curated agent-only dataset: examples
of the `execute_code` tool call format where the model produces code, an executor runs it,
and the observation confirms the output. This training teaches the tool-call format and
basic code-generation habits without injecting execution traces that shift the model's
output distribution. The adapter is merged at inference time using `merge_and_unload`,
verified safe (§5.1). All results are obtained on a single RTX 4080 Super (16 GB VRAM)
with no cloud inference. See Appendix B for hardware details.

### 3.2 Memory System

The memory is an offline verified vector store. Each record contains a task description
(`query_text`), a verified `execute_code` tool call (`corrected_tool_call`), and a
confirmed execution result (`PASS`). A local TF-IDF-like sentence embedder
(`code-memory-embedder`) generates dense vectors; retrieval uses cosine similarity with
k=4 (ablation-selected, §5.1). Retrieved records are formatted as a
`RETRIEVED_VERIFIED_MEMORY` block in the system prompt; they are guidance only — the
model must produce its own tool call.

Two indexes are used in this study. **The champion index** (`memory/index_adapted`,
99 records) is evaluated on all clean benchmarks and is the only configuration reported
as the clean held-out champion. **The repair index** (`memory/index_adapted_v29`,
103 records) adds 4 targeted repair records; it is evaluated diagnostically and on clean
generalisation tasks, but is never reported as the clean champion because it was built
after the frozen benchmark was established.

### 3.3 Evaluation Protocol

**Benchmark.** 28 hard held-out tasks covering single-function Python coding problems
with assertion-based verification. Frozen before any experiment: not used in training,
hyperparameter search, or repair targeting until after the clean champion score was
established.

**Scoring.** best-of-3 sampling, verified_agent scoring. The model must produce an
`execute_code` tool call whose execution returns `PASS`. Fallback scoring disabled.
Strict agent contract: no direct answer extraction.

**Variance.** Approximately ±2–3 tasks on the 32-task benchmark at best-of-3. Differences
of fewer than 3 tasks cannot be attributed to the experimental condition rather than
sampling noise.

---

## 4. Experimental Setup

All experiments use the same frozen 28-task benchmark except v2.10 and v2.11, which
additionally use a 32-task clean generalisation benchmark (zero overlap with frozen tasks,
constructed after all repair records were finalised). No experiment modifies the champion
model weights after the v2.7 audit.

**Contamination rule.** Any benchmark that receives repair records targeting known
failures is immediately reclassified from clean to diagnostic. Results on that benchmark
are not reported as the clean held-out champion. This rule applies from the moment the
repair index was constructed; all 27/28 = 96.4% results are therefore diagnostic.

Experiments are run sequentially; each result informs the next experiment design. All
experiments are reported, including failures. The negative results are as informative as
the positive ones.

---

## 5. Results

See [Table 1] in `paper/tables/main_results_table.md` for the complete summary across all
configurations. See [Figure 1] in `paper/figures/experiment_timeline_spec.md` for the
full experiment timeline.

### 5.1 Memory Is Load-Bearing (v2.7)

**Setup.** Evaluate the champion adapter with and without memory at varying k values
(see [Table 3] in `paper/tables/main_results_table.md`).

| Configuration | Score | Δ vs champion |
|---|---|---|
| Adapter merged + memory k=4 | **23/28 = 82.1%** | — |
| Adapter merged + memory k=5 | 22/28 = 78.6% | −3.5 pp |
| Adapter unmerged + memory k=4 | 22/28 = 78.6% | −3.5 pp |
| Adapter merged + memory k=1 | 20/28 = 71.4% | −10.7 pp |
| Adapter merged, no memory | 18/28 = 64.3% | **−17.8 pp** |

**Memory lift: +17.8 pp.** Memory retrieval is not passive — it is required for 6 tasks
that fail entirely without retrieved context (count_islands, gcd_lcm, simple_ttl_cache,
topo_sort, tree_depth_tuple, valid_ipv4). These span diverse pattern categories (graph
traversal, caching, IP validation), indicating the lift is broad rather than concentrated
in one algorithm family. Two tasks are *hurt* by memory (merge_intervals, slugify) due to
retrieval noise (§6.1).

**Merge safety.** Merged and unmerged adapters differ by ≤3.5 pp (within variance).
`merge_and_unload` is safe for this model.

**k=4 is optimal.** k=1 loses 10.7 pp versus k=4; k=5 loses 3.5 pp. Four retrieved
examples provide sufficient context without diluting the prompt with less relevant records.

### 5.2 Retraining Does Not Recover or Improve the Champion (v2.5 / v2.6)

**Setup.** Fine-tune the merged champion with additional data at varying trace ratios
and learning rates (see [Table 2] in `paper/tables/main_results_table.md`).

| Experiment | LR | Score | Δ vs champion |
|---|---|---|---|
| Merge + fresh LoRA | 5e-6 | 64.3% | −17.9 pp |
| v2.5 clean foundation (5282 ex.) | 2e-5 | 53.6% | −21.4 pp |
| v2.6 traces=0% (2282 ex.) | 2e-5 | 57.1% | −17.9 pp |
| v2.6 traces=10% | 2e-5 | 50.0% | −25.0 pp |
| v2.6 traces=25% | 2e-5 | 53.6% | −21.4 pp |

All five experiments cause regressions. Training loss was nearly identical across v2.6
runs (0.129–0.132), ruling out training instability. The pattern holds across data sizes,
trace ratios, and learning rates.

**Root cause.** The original 300-step, 6e-6 LR, agent-only training produced an adapter
whose generalisation depends on the specific trajectory. Adding more data at 2e-5 LR
overwrites this property: hard task performance recovers (+24 pp) but string task
performance collapses (−50 pp) and basic tasks fail entirely (−100 pp). The mechanism is
output distribution shift from direct-answer format to trace-generating format.

**Conclusion.** Further retraining is not a viable path. The champion adapter is frozen.

### 5.3 Targeted Repair Memory — Diagnostic Only (v2.9)

> **This result is diagnostic, not a clean held-out champion.** The frozen benchmark is
> no longer independent for this evaluation: repair records target the exact 4 failing tasks.

**Setup.** Identify the 4 failing tasks, trace their retrieval noise, and create 4 verified
repair examples. Evaluate on the frozen benchmark to confirm the repair records have the
intended effect (see [Table 4] in `paper/tables/main_results_table.md`).

| Task | Champion | Repair (diagnostic) | Note |
|---|---|---|---|
| merge_intervals | FAIL | PASS | Valid repair |
| median_two_sorted | FAIL | PASS | Valid repair |
| deep_get | FAIL | PASS | Valid repair |
| tree_depth_tuple | FAIL | PASS | Spec-conflicted (§6.2)† |

†The prompt contains a broken assertion; PASS via repair memory means the model followed
the repair record's correct assertion, not the task prompt.

**Diagnostic result: 27/28 = 96.4%.** This demonstrates that the right context fixes the
failures; it does not establish that the system has genuinely improved on held-out data.
Do not compare this number to the clean champion score (23/28 = 82.1%).

**Early transfer signal.** A 5-task clean transfer test (tasks similar to repair targets
but not identical) reached 4/5 = 80%. Positive signal, but too small for a strong claim;
the 32-task benchmark in §5.4 provides the appropriate scale.

### 5.4 Repair Memory Does Not Generalise at Scale — Rejected (v2.10)

**Setup.** 32 untouched tasks across 5 algorithm families, zero overlap with the frozen
28-task benchmark or v2.9 tasks (see [Table 5] in `paper/tables/main_results_table.md`).

| Family | N | Champion | Repair | Δ |
|---|---|---|---|---|
| Interval merging | 6 | 4/6 = 66.7% | 5/6 = 83.3% | +16.7 pp |
| Sorted-array / kth | 7 | 3/7 = 42.9% | 3/7 = 42.9% | 0 pp |
| Nested dict | 7 | 6/7 = 85.7% | 5/7 = 71.4% | **−14.3 pp** |
| Tuple-tree recursion | 7 | 3/7 = 42.9% | 2/7 = 28.6% | **−14.3 pp** |
| Run-length encoding | 5 | 4/5 = 80.0% | 3/5 = 60.0% | **−20.0 pp** |
| **Total** | **32** | **20/32 = 62.5%** | **18/32 = 56.2%** | **−6.3 pp** |

The repair index gains 2 tasks (`interval_union`, `running_median`) and loses 4 others
(`deep_delete`, `unflatten_dict`, `tree_mirror`, `merge_sorted_k`), for a net change of
−2 tasks (−6.3 pp). The interval_merge family is the only one where repair helps
(+16.7 pp), consistent with those repair records targeting interval-specific patterns.
The three regressing families all share vocabulary with repair records not designed for
them — the mechanism is repair vocabulary leak (§6.1, Type 3).

**Root cause.** Adding repair records to the full index changes the retrieval landscape
for all 32 tasks. The `deep_get` repair record scores 0.64–0.82 against all 7 nested_dict
tasks; the `tree_depth` repair record scores 0.59–0.72 against all 7 tuple_tree tasks.
These repair records become the nearest neighbour for tasks they were not designed to fix.

**Conclusion.** Global repair-index promotion is rejected. The champion index (99 records)
is the superior configuration for clean generalisation.

### 5.5 TF-IDF Routing Cannot Selectively Gate Repair Memory — Rejected (v2.11)

**Setup.** Three routing strategies tested; champion index is the baseline at 20/32 = 62.5%
(see [Table 6] in `paper/tables/main_results_table.md`).

**Key constraint.** Only 3 of 32 tasks benefit from repair routing under any routing
scheme; 9 tasks fail with both indexes and are unreachable regardless of which index is
chosen. The oracle ceiling is 23/32 = 71.9% — diagnostic and not achievable by any
deployed router.

| Strategy | Tasks → Repair | Score | Δ vs champion |
|---|---|---|---|
| Family router | 6 (interval_merge family) | 19/32 = 59.4% | −3.1 pp |
| Confidence router (TF-IDF margin ≥ 0.05) | 20 | 20/32 = 62.5% | 0 pp |
| **Oracle ceiling** (diagnostic, not deployable) | **3** | **23/32 = 71.9%** | **+9.4 pp** |

**Confidence router.** Routes to repair when repair top-1 TF-IDF score ≥ 0.35 AND
beats champion top-1 by ≥ 0.05. This fires on 20 of 32 tasks — the threshold is too
permissive, not too strict. Three gains (`interval_union`, `running_median`, `deep_keys`)
exactly cancel three losses (`deep_delete`, `unflatten_dict`, `tree_mirror`): tied at 20/32.

**Family router.** Routes all 6 interval_merge tasks to repair. The single expected gain
(`interval_union`) was not observed in this run due to sampling variance. Net: −3.1 pp.

**Root cause.** TF-IDF similarity measures vocabulary-family overlap, not
algorithm-specific repair relevance. The signal fires at family granularity (scoring
0.59–0.82 against entire families) when it needs to fire at task granularity (3 specific
tasks in 32). This is not a tuning problem; it is a fundamental limit of bag-of-words
similarity for routing decisions (see §6 for the full taxonomy).

---

## 6. Failure Analysis

All five negative results — retraining, global repair, family routing, confidence routing,
and the routing gap below oracle ceiling — are explained by the same root cause: TF-IDF
similarity measures token frequency, not algorithm structure. An embedder that encodes
what a function does rather than what words it uses would resolve all three of the
following failure types. See [Figure 2] in `paper/figures/retrieval_failure_taxonomy_spec.md`
for the taxonomy diagram specification.

### 6.1 Retrieval Failure Taxonomy

**Type 1 — Surface lexical collision.** High-frequency tokens bridge algorithmically
unrelated records despite different intent. `merge_intervals` retrieves `merge_sorted`
(top-1 similarity 0.557) via tokens "merge" and "sorted." Eight `merge_sorted` records
dominate k=4 retrieval, injecting a sort-and-merge pattern where an interval-sweep
pattern is needed.

**Type 2 — Structural vocabulary overlap.** Nested-data tasks share structural tokens
with unrelated nested-pattern records. `tree_depth_tuple` retrieves `flatten` records
via "nested", "list", "depth"; `deep_get` retrieves `invert_dict` and `flatten` via
"dict", "nested", "key." The retriever cannot distinguish "this task involves nested
structures" from "this task requires this specific nested-structure algorithm."

**Type 3 — Repair vocabulary leak.** Repair records are densely written with
algorithm-family vocabulary to ensure they retrieve correctly for the target task.
This causes them to score highly against all tasks in the same family. The `deep_get`
repair record scores 0.64–0.82 against all 7 nested_dict tasks; the `tree_depth` repair
record scores 0.59–0.72 against all 7 tuple_tree tasks. Any routing heuristic that uses
TF-IDF similarity to identify repair targets will fire at family granularity rather than
task granularity.

All three types share the same root cause: **bag-of-words similarity conflates algorithm
identity with lexical co-occurrence.** Types 1 and 2 explain failures with the champion
index; Type 3 explains why global repair promotion and TF-IDF routing both fail.

### 6.2 Benchmark Integrity Finding

The `tree_depth_tuple` task contains a spec-conflicted assertion. Forensic analysis
beyond aggregate scores reveals a task that consistently fails despite correct
implementations — this indicates a benchmark defect, not a model defect.

The prompt expects:

```
tree_depth(((1,2),(3,(4,5)))) == 3
```

By the stated rule (leaves have depth 1; a branch has depth 1 + max(children)):

```
tree_depth((4,5))            = 2
tree_depth((3,(4,5)))        = 3
tree_depth(((1,2),(3,(4,5)))) = 1 + max(2,3) = 4
```

The correct value is **4**, not 3. Verified computationally. A correct implementation
produces 4, which fails the assertion == 3.

**Action taken.** Documented as a spec-conflicted benchmark defect. All results are
reported both raw (including the defective task) and as a corrected audit score
(excluding it). Raw champion: 23/28 = 82.1%. Corrected champion: 24/27 = 88.9%.
The repair record uses the correct assertion (== 4); the task PASS with the repair index
reflects model compliance with the repair record, not the task prompt.

### 6.3 Sampling Variance

At best-of-3 sampling, individual task outcomes vary between runs. Tasks observed to
flip across sub-evaluations: `interval_union`, `insert_interval`, `merge_sorted_k`,
`deep_delete`, `rle_expand`. This variance (±2–3 tasks on a 32-task benchmark) means
that a difference of fewer than 3 tasks cannot be attributed to the experimental
condition rather than sampling noise. Results near the 62.5% champion baseline on the
32-task benchmark should be interpreted within this bound.

### 6.4 Tasks at the Model Capability Ceiling

Nine of 32 clean tasks fail with both the champion and repair indexes in v2.10:
`kth_smallest_matrix`, `wiggle_sort`, `count_smaller_after_self`, `tree_max_path_sum`,
`tree_from_list`, `tree_serialize`, `tree_width`, `rle_delta_encode`, and `insert_interval`
(most runs). These represent the model capability ceiling — tasks requiring complex
multi-step reasoning or algorithms beyond the current memory patterns — not retrieval
failures. Better retrieval cannot recover them.

---

## 7. Limitations

**Scale.** The frozen benchmark contains 28 tasks; the generalisation benchmark contains
32 tasks. At best-of-3 sampling, variance is ±2–3 tasks. The benchmarks were constructed
to be hard (single-function, assertion-verified, diverse algorithm categories), so the
absolute scores are not directly comparable to self-reported results on easier benchmarks.
Effect sizes below 3 tasks should be interpreted as within noise.

**Comparisons.** No baseline comparison to other code-generation systems has been run on
these benchmarks. Any comparative claim requires co-evaluation on the same tasks under
the same protocol.

**Task distribution.** All evaluation tasks are hard single-function Python coding
problems. Generalisation to multi-file tasks, natural language QA, or real-world
repository bugs has not been tested.

**Retrieval architecture.** All results use TF-IDF-like embedding. Dense semantic
retrieval (CodeBERT, UniXcoder) was not evaluated; the hypothesis that algorithm-aware
embeddings would resolve the failure modes is untested.

**Model family.** All results use a single base model (Qwen2.5-Coder-1.5B-Instruct).
Whether the memory lift and retrieval failure modes transfer to other architectures or
parameter scales is untested.

**SWE-bench.** The system has not been evaluated on any real-world repository patch task.
No SWE-bench claims are made.

---

## 8. Reproducibility

All results were obtained on a single RTX 4080 Super (16 GB VRAM) with no cloud inference
or external API. The champion model must be present at
`outputs/qwen15b_v27_champion_merged`.

**Core commands:**

```bash
make eval-champion              # 23/28 = 82.1% clean champion
make eval-v210-clean-champion   # 20/32 = 62.5% clean generalisation
make eval-v210-repair-index     # 18/32 = 56.2% repair comparison
make summarise-v211             # Routing audit summary
```

Full command listing, environment setup, and hardware requirements: see
Appendix A and `results/v212_manuscript_packet/05_reproducibility_commands.md`.

**Sampling variance.** The champion 23/28 result is stable across repeated runs.
Clean-benchmark scores at 62.5% should be reported with ±2–3 task uncertainty.

---

## 9. Future Work

**Dense code retrieval.** Replacing TF-IDF with a code-specific semantic embedder
(CodeBERT, UniXcoder, nomic-embed-code) is the most direct path to addressing all three
failure types. The oracle ceiling (23/32 = 71.9%) provides a concrete target: an accurate
retrieval signal would recover 3 additional tasks. Dense retrieval does not guarantee
reaching the oracle ceiling, but it removes the vocabulary-overlap mechanism that blocks
all three current failure types.

**Operation-aware memory metadata.** Tagging each memory record with algorithm family
labels would allow exact-match family routing without TF-IDF false positives. The v2.11
oracle analysis identifies which 3 tasks benefit; at least 2 of them (`interval_union`
via interval_merge family; `deep_keys` via nested_dict family) could be correctly routed
by an exact-match classifier.

**SWE-bench patch generation.** The evaluation harness covers single-function Python
tasks. Extending to SWE-bench Lite would test whether memory-assisted retrieval helps
on real multi-file repository tasks with established comparison baselines.

**Larger benchmark.** A 200+ task benchmark would reduce sampling variance to below ±1
task and support statistical effect-size estimates. Current results are reliable at the
5-task granularity of the memory lift finding, but marginal for 2–3 task routing
differences.

---

## 10. Conclusion

Verified offline memory retrieval is load-bearing for a local code-agent based on
Qwen2.5-Coder-1.5B-Instruct: removing the memory index drops performance by 17.8
percentage points. Five controlled follow-up experiments show that none of the tested
improvement strategies — retraining, global repair-memory promotion, or TF-IDF confidence
routing — improves over the champion configuration on clean generalisation tasks.

The binding constraint is retrieval relevance. TF-IDF similarity measures vocabulary
co-occurrence, not algorithm-level similarity. This causes three distinct failure modes:
lexical collision between algorithmically unrelated records, structural vocabulary overlap
across nested-data patterns, and repair vocabulary leak across algorithm families. All
three share the same root cause and point to the same solution: an embedding that encodes
algorithm structure rather than token frequency.

The oracle analysis separates the retrieval bottleneck (3 of 32 tasks recoverable by
better routing) from the model capability ceiling (9 tasks failing regardless of routing).
Addressing the full oracle ceiling requires both better retrieval and broader model
capability.

The clean champion — 23/28 = 82.1% on the frozen benchmark, fully reproducible on a
single consumer GPU — is the stable result for this arc. It establishes a concrete
baseline for future retrieval-improvement experiments and identifies retrieval relevance
as the next engineering target.

---

## Appendix

### Appendix A — Full Reproducibility Commands

```bash
# Environment
conda activate aetherforge-train

# Champion baseline (v2.7)
make eval-champion
make summarise-v27-preservation

# Retraining audit (v2.5 / v2.6)
make summarise-v26

# Hyperparameter audit (v2.8)
make summarise-v28

# Repair memory diagnostic (v2.9)
make eval-v29-repair-memory-diagnostic
make summarise-v29

# Clean generalisation (v2.10)
make eval-v210-clean-champion
make eval-v210-repair-index
make summarise-v210

# Routing audit (v2.11)
make route-v211
make eval-v211-family-router
make eval-v211-confidence-router
make summarise-v211
```

All targets require the champion model at `outputs/qwen15b_v27_champion_merged`.
See `results/v212_manuscript_packet/05_reproducibility_commands.md` for full details
including environment setup and expected runtimes.

### Appendix B — Hardware

| Component | Specification |
|---|---|
| GPU | RTX 4080 Super (16 GB VRAM) |
| Model VRAM (merged, fp16) | ~3.2 GB |
| Total VRAM during eval | ~5–7 GB |
| Inference batch size | 1 |
| Cloud inference | None |

### Appendix C — Supporting Documents

| Document | Path |
|---|---|
| Full results tables | `paper/tables/main_results_table.md` |
| Experiment timeline figure spec | `paper/figures/experiment_timeline_spec.md` |
| Retrieval failure taxonomy figure spec | `paper/figures/retrieval_failure_taxonomy_spec.md` |
| Experiment timeline (narrative) | `results/v212_manuscript_packet/01_experiment_timeline.md` |
| Full failure analysis | `results/v212_manuscript_packet/03_failure_analysis.md` |
| Claim boundary | `results/v212_manuscript_packet/04_claim_boundary.md` |
| Per-task routing scores (v2.11) | `results/v211_retrieval_routing/per_task_routing.csv` |
| Evidence closure certificate | `docs/evidence_closure_certificate.md` |
| Reviewer claim boundary | `paper/reviewer_claim_boundary.md` |
