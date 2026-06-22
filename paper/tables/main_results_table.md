# Table 1 — Core System Performance

| Configuration | Benchmark | Score | Type |
|---|---|---|---|
| Adapter-only (no memory) | Frozen 28-task | 18/28 = 64.3% | Clean |
| Adapter + champion memory (k=4) | Frozen 28-task | **23/28 = 82.1%** | **Clean champion** |
| Adapter + repair index (diagnostic) | Frozen 28-task | 27/28 = 96.4% | Diagnostic only† |
| Adapter + champion memory | 32 clean tasks | 20/32 = 62.5% | Clean |
| Adapter + repair index | 32 clean tasks | 18/32 = 56.2% | Rejected |

†The frozen benchmark is no longer independent for this configuration because repair records target known failures.

---

# Table 2 — Retraining Experiments (all rejected)

| Experiment | LR | Steps | Score | vs Champion |
|---|---|---|---|---|
| Original adapter (champion) | 6e-6 | 300 | 75–78.6% | baseline |
| Merge + fresh LoRA | 5e-6 | 350 | 64.3% | −17.9 pp |
| v2.5 clean foundation (5282 ex.) | 2e-5 | 300 | 53.6% | −21.4 pp |
| v2.6 traces=0% (2282 ex.) | 2e-5 | 300 | 57.1% | −17.9 pp |
| v2.6 traces=10% (2536 ex.) | 2e-5 | 300 | 50.0% | −25.0 pp |
| v2.6 traces=25% (3043 ex.) | 2e-5 | 300 | 53.6% | −21.4 pp |

---

# Table 3 — Memory Ablation (v2.7)

| Configuration | Score | Delta |
|---|---|---|
| Adapter merged + memory k=4 | 23/28 = 82.1% | — |
| Adapter merged + memory k=1 | 20/28 = 71.4% | −10.7 pp |
| Adapter merged + memory k=5 | 22/28 = 78.6% | −3.5 pp |
| Adapter merged, no memory | 18/28 = 64.3% | −17.8 pp |
| Adapter unmerged + memory k=4 | 22/28 = 78.6% | −3.5 pp |

---

# Table 4 — Repair Memory (v2.9 diagnostic)

| Task | Pattern type | Champion | Repair | Note |
|---|---|---|---|---|
| merge_intervals | Interval merge | FAIL | PASS | Valid repair |
| median_two_sorted | Two-pointer merge | FAIL | PASS | Valid repair |
| deep_get | Nested dict access | FAIL | PASS | Valid repair |
| tree_depth_tuple | Recursive tree | FAIL | PASS | Spec-conflicted† |

†Task prompt contains broken assertion (==3; correct is ==4). PASS via repair memory means model followed correct assertion from repair record.

---

# Table 5 — Clean Generalisation per Family (v2.10, 32 tasks)

| Family | N | Champion | Repair | Δ |
|---|---|---|---|---|
| Interval merging and scheduling | 6 | 4/6 = 66.7% | 5/6 = 83.3% | +16.7 pp |
| Sorted-array / median / kth | 7 | 3/7 = 42.9% | 3/7 = 42.9% | 0 pp |
| Nested dictionary access/update | 7 | 6/7 = 85.7% | 5/7 = 71.4% | −14.3 pp |
| Tuple-tree recursion | 7 | 3/7 = 42.9% | 2/7 = 28.6% | −14.3 pp |
| Run-length encoding | 5 | 4/5 = 80.0% | 3/5 = 60.0% | −20.0 pp |
| **Total** | **32** | **20/32 = 62.5%** | **18/32 = 56.2%** | **−6.3 pp** |

---

# Table 6 — Routing Audit (v2.11, 32 tasks)

| Strategy | Description | Score | vs Champion |
|---|---|---|---|
| Champion index | Baseline | 20/32 = 62.5% | — |
| Repair index | Rejected (v2.10) | 18/32 = 56.2% | −6.3 pp |
| Family router | interval_merge → repair, others → champion | 19/32 = 59.4% | −3.1 pp |
| Confidence router | Repair when TF-IDF margin ≥ 0.05 | 20/32 = 62.5% | 0 pp |
| Oracle ceiling | Perfect per-task routing (diagnostic) | 23/32 = 71.9% | +9.4 pp |

Oracle breakdown: 15 tasks both-pass, 5 champion-only, 3 repair-only, 9 both-fail.
