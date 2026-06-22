# AetherForge — Experiment Timeline

## Setup (pre-v2.6)

- Base model: Qwen2.5-Coder-1.5B-Instruct
- LoRA training: 300 steps, 6e-6 LR, agent-only curated data (execute_code tool use)
- Frozen 28-task benchmark: hard held-out tasks, never used in training
- Original adapter eval: 75.0–78.6% (sampling variance across runs, best-of-3)

## v2.6 — Trace-Gating Ablation (NEGATIVE)

**Goal:** Test whether adding execution traces to training data recovers a higher score.

| Trace ratio | Steps | Score | vs champion |
|---|---|---|---|
| 0% traces | 300 | 57.1% | −17.9 pp |
| 10% traces | 300 | 50.0% | −25.0 pp |
| 25% traces | 300 | 53.6% | −21.4 pp |

**Finding:** Execution traces are harmful at every tested dose. Training loss was
nearly identical across runs (0.129–0.132), ruling out instability. The original
adapter's generalisation properties depend on its training trajectory and cannot
be recovered through merge-and-retrain.

## v2.7 — Champion Preservation Audit (POSITIVE)

**Goal:** Verify that merge_and_unload is safe and measure the memory contribution.

| Configuration | Score |
|---|---|
| Adapter unmerged, with memory | 22/28 = 78.6% |
| Adapter merged, with memory | 23/28 = 82.1% ← champion |
| Adapter merged, NO memory | 18/28 = 64.3% |

**Finding:** Merge is safe (+3.5 pp within noise). Memory lift = +17.8 pp.
6 tasks pass only with memory: count_islands, gcd_lcm, simple_ttl_cache,
topo_sort, tree_depth_tuple, valid_ipv4.
2 tasks are hurt by memory: merge_intervals, slugify (retrieval noise).

## v2.8 — Champion System Enhancement (NEGATIVE)

**Goal:** Find a hyperparameter or prompt variant that beats 23/28.

| Variant | Score |
|---|---|
| Baseline k=4 | 23/28 = 82.1% |
| k=1 | 20/28 = 71.4% |
| k=5 | 22/28 = 78.6% |
| Filtered index (83 rec) | 23/28 = 82.1% |
| Direct-answer prompt | 21/28 = 75.0% |

**Finding:** No variant beat champion. k=4 is optimal; filtering does not help.
Retrieval noise identified: `merge_sorted` records contaminate `merge_intervals`;
`flatten` records contaminate `tree_depth_tuple`; `deep_get` and `median_two_sorted`
are structural failures unrelated to noise.

## v2.9 — Memory Repair Split (DIAGNOSTIC)

**Goal:** Test targeted repair records for the 4 failing tasks without contaminating
the champion index.

| Lane | Score | Label |
|---|---|---|
| Champion index (99 rec) | 23/28 = 82.1% | Clean champion (unchanged) |
| Repair diagnostic (103 rec) | 27/28 = 96.4% | Diagnostic — NOT clean champion |
| Corrected audit (excl. tree_depth_tuple) | 26/27 = 96.3% | Diagnostic |
| Clean generalisation (5 tasks) | 4/5 = 80.0% | External signal |

**Additional finding:** `tree_depth_tuple` task prompt contains a broken assertion
(`tree_depth(((1,2),(3,(4,5)))) == 3` should be 4 by stated rule). Confirmed
computationally. Documented as spec defect; not silently removed from benchmark.

**Finding:** Repair memory can fix known failures when retrieved. The 4/5 = 80%
clean transfer provided an early positive signal that later did not hold at scale.

## v2.10 — Clean Repair-Generalisation Benchmark (NEGATIVE)

**Goal:** 32 untouched tasks, 5 families; test whether repair memory generalises.

| Lane | Score |
|---|---|
| Champion index | 20/32 = 62.5% |
| Repair index | 18/32 = 56.2% |

**Per-family:** interval_merge +16.7 pp (repair helps); nested_dict −14.3 pp,
tuple_tree −14.3 pp, rle_encoding −20.0 pp (regressions).

**Finding:** Adding repair records globally reshuffles retrieval for ALL tasks,
not just repair targets. Net −2 tasks. Global repair-index promotion rejected.

## v2.11 — Retrieval Routing and Gating Audit (NEGATIVE)

**Goal:** Test selective routing to preserve interval_merge gain without regressions.

| Router | Score | vs Champion |
|---|---|---|
| Family router (interval_merge → repair) | 19/32 = 59.4% | −3.1 pp |
| Confidence router (TF-IDF margin ≥ 0.05) | 20/32 = 62.5% | 0 pp |
| Oracle ceiling | 23/32 = 71.9% | +9.4 pp |

**Finding:** TF-IDF similarity margin measures vocabulary family overlap, not repair
relevance. repair records for `deep_get` and `tree_depth` score 0.59–0.82 against
ALL nested_dict and tuple_tree tasks. Only 3 tasks benefit from per-task routing.
Routing gains (±3 tasks) are within best-of-3 sampling variance.

## Summary

Five controlled experiments, all with clear causal findings. One stable positive result
(memory lift), four negative results with identified root causes. The champion is frozen
at 23/28 = 82.1%. The binding constraint is retrieval relevance, not memory quantity.
