# AetherForge: Memory-Augmented Code-Agent Adaptation
## Two-Page Research Summary

**Frank Asante Van Laarhoven**

---

### Problem

Small language models fine-tuned for local code generation face two competing pressures.
On one hand, retraining on more data or longer schedules risks overwriting the specific
generalisation trajectory that makes the base adapter useful. On the other hand, without
any form of contextual guidance, small models lack the in-context examples that larger
models can retrieve from their parametric memory. Offline verified retrieval — a local
vector store of confirmed working solutions — offers a middle path: provide tested examples
at inference time without changing model weights. The question is whether this retrieval
is genuinely load-bearing, and whether augmenting or routing across multiple indexes can
improve it further.

---

### System

AetherForge uses Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA (rank 16, alpha 32,
300 steps, learning rate 6e-6) on a curated agent-only dataset. The adapter is merged at
inference time. A 99-record offline verified memory is retrieved via TF-IDF cosine
similarity (top-4 records). The model generates Python functions as `execute_code` tool
calls; a verifier confirms correctness by assertion. All experiments use best-of-3
sampling with strict verified-agent scoring on a consumer GPU (RTX 4080 Super, 16 GB
VRAM). No cloud API is used at any stage.

---

### Results

**Table 1. Core results across all experiments.**

| Configuration | Benchmark | Score | Type |
|---|---|---|---|
| Adapter, no memory | Frozen 28-task | 18/28 = 64.3% | Clean |
| **Champion: adapter + memory (k=4)** | **Frozen 28-task** | **23/28 = 82.1%** | **Clean** |
| Repair index (diagnostic) | Frozen 28-task | 27/28 = 96.4% | **Diagnostic†** |
| Champion on generalisation | 32 clean tasks | 20/32 = 62.5% | Clean |
| Repair on generalisation | 32 clean tasks | 18/32 = 56.2% | Rejected |
| Oracle routing ceiling | 32 clean tasks | 23/32 = 71.9% | Diagnostic |

†The 96.4% result targets known frozen-benchmark failures. The benchmark is not
independent for this configuration and is not reported as the clean champion.

---

**Memory is load-bearing.** Removing the 99-record index costs 17.8 percentage points
(82.1% → 64.3%). This is the project's primary positive finding. The ablation confirms
k=4 as optimal; k=1 loses 10.7 pp, and the LoRA merge is verified safe (≤3.5 pp delta).

**Retraining is harmful.** Five training configurations were tested — varying data mix,
trace ratio, and learning rate. All regressed versus the champion, by 17.9–25.0 pp. The
root cause is output distribution shift: higher learning rates recover hard-task
performance (+24 pp) but collapse string-task performance (−50 pp) and basic tasks
(−100 pp). The original 300-step, 6e-6 LR, agent-only trajectory is not recoverable by
continued training.

**Targeted repair fixes known failures, but does not generalise.** Four repair records
targeting the four frozen-benchmark failures produce 27/28 = 96.4% diagnostically. When
the same repair index is evaluated on 32 fresh tasks, it loses to champion: 56.2% vs
62.5%. Net task flip: +2 gained (interval_union, running_median), −4 lost (deep_delete,
unflatten_dict, tree_mirror, merge_sorted_k). Repair records reshuf-fle TF-IDF retrieval
for the entire index, not just their intended targets.

**No routing strategy improves over champion.** Family routing (59.4%), TF-IDF
confidence routing (62.5%, tied), and oracle analysis (71.9%, diagnostic) were all
tested. The oracle analysis shows only 3 of 32 tasks benefit from repair routing; 9 tasks
fail with both indexes. The confidence router fires on 20 of 32 tasks because TF-IDF
repair records score 0.59–0.82 against entire algorithm families, not just repair
targets.

---

### Failure Taxonomy

Three retrieval failure classes were identified, all sharing the same root cause.

**Type 1 — Surface lexical collision.** High-frequency tokens bridge algorithmically
unrelated records. `merge_intervals` retrieves `merge_sorted` (similarity 0.557) via
shared tokens "merge" and "sorted."

**Type 2 — Structural vocabulary overlap.** Nested-data tasks share vocabulary with
unrelated nested-pattern records. `tree_depth_tuple` retrieves `flatten` records via
"nested", "list", "depth."

**Type 3 — Repair vocabulary leak.** Repair records are dense with family vocabulary,
causing them to score highly against all tasks in the family. The `deep_get` repair
record scores 0.64–0.82 against all 7 nested-dict tasks; the `tree_depth` repair record
scores 0.59–0.72 against all 7 tuple-tree tasks.

Root cause: **TF-IDF similarity ≠ algorithm-level similarity.** Bag-of-words retrieval
cannot distinguish "this task requires this pattern" from "this task shares vocabulary
with this family."

---

### Benchmark Integrity

One frozen-benchmark task (`tree_depth_tuple`) contains a spec-conflicted assertion: the
prompt expects depth 3, but the stated recursive rule gives 4. A computationally correct
implementation fails the assertion. This was discovered during v2.9, reported
transparently, and documented. The raw conservative champion score is 23/28 = 82.1%;
the corrected audit score excluding the defective task is 24/27 = 88.9%.

---

### Limitations

Benchmark size: 28 frozen tasks and 32 generalisation tasks. At best-of-3 sampling,
variance is ±2–3 tasks; gains below 3 tasks cannot be reliably distinguished from noise.
No comparison to larger models or other code-generation systems was run on these
benchmarks. Results apply to single-function Python coding tasks with assertion-based
verification. Multi-file repository patch generation was not evaluated.

---

### Conclusion

Verified memory retrieval is load-bearing for this code-agent (+17.8 pp). Retraining,
global repair-memory promotion, and TF-IDF routing do not improve over the champion on
clean generalisation tasks. The binding constraint is retrieval relevance: TF-IDF
similarity measures vocabulary co-occurrence rather than algorithmic identity. The oracle
ceiling (71.9%) separates the retrieval bottleneck (3 recoverable tasks) from the model
capability ceiling (9 tasks failing regardless of index). The clean champion — 23/28 =
82.1%, fully reproducible on a single GPU — is the stable result for this arc.

---

### Future Work

**Dense code retrieval.** Replace TF-IDF with a code-specific semantic embedder
(CodeBERT, UniXcoder, nomic-embed-code) to address all three retrieval failure types.
This is the single highest-leverage next step: the oracle ceiling shows 3 additional
recoverable tasks if routing becomes accurate.

**Operation-aware metadata.** Tag each memory record with algorithm family labels for
exact-match routing, bypassing TF-IDF false positives for at least 2 oracle-recoverable
tasks (interval_union via interval_merge family; deep_keys via nested_dict family).

**External benchmark.** SWE-bench Lite patch generation to test whether memory-assisted
retrieval helps on real repository tasks with multi-file context.

**Larger evaluation set.** 200+ tasks to reduce sampling variance to below ±1 task and
support statistical effect-size estimates.
