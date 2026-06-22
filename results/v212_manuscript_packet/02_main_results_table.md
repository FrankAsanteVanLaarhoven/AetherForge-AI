# AetherForge — Main Results Table

## Core system result

| Configuration | Benchmark | Score | Type |
|---|---|---|---|
| Adapter-only (no memory) | Frozen 28-task | 18/28 = 64.3% | Clean |
| Adapter + champion index (k=4) | Frozen 28-task | 23/28 = 82.1% | **Clean champion** |
| Adapter + repair index (diagnostic) | Frozen 28-task | 27/28 = 96.4% | Diagnostic only |
| Adapter + champion index | 32 clean tasks | 20/32 = 62.5% | Clean |
| Adapter + repair index | 32 clean tasks | 18/32 = 56.2% | Rejected |

## Memory ablation (v2.7)

| Configuration | Score | Delta |
|---|---|---|
| Champion model, memory k=4 | 23/28 = 82.1% | baseline |
| Champion model, no memory | 18/28 = 64.3% | −17.8 pp |
| Champion model, k=1 | 20/28 = 71.4% | −10.7 pp |
| Champion model, k=5 | 22/28 = 78.6% | −3.5 pp |

## Retraining experiments (v2.5 / v2.6) — all rejected

| Experiment | Score | Failure cause |
|---|---|---|
| Targeted LoRA pilots (v2.4) | 57.1% | Adapter stacking conflict |
| Merge + fresh LoRA, 350 steps | 64.3% | Capability regression |
| v2.5 clean foundation (2e-5 LR, 5282 ex.) | 53.6% | Data mixture: traces harm string tasks |
| v2.6 traces=0% (2e-5 LR, 2282 ex.) | 57.1% | Best retraining result, still −17.9 pp |
| v2.6 traces=10% | 50.0% | Worse than 0% |
| v2.6 traces=25% | 53.6% | Same as v2.5 |

## Repair memory (v2.9)

| Lane | Score | Label |
|---|---|---|
| Champion (baseline) | 23/28 = 82.1% | Clean, frozen |
| Repair diagnostic (103 rec) | 27/28 = 96.4% | Diagnostic — NOT clean |
| Corrected (excl. spec-conflicted) | 26/27 = 96.3% | Diagnostic |
| v2.9 clean transfer (5 tasks) | 4/5 = 80.0% | Early external signal |

### Repair target outcomes (v2.9 diagnostic)

| Task | Champion | Diagnostic | Type |
|---|---|---|---|
| merge_intervals | FAIL | PASS | Valid repair |
| median_two_sorted | FAIL | PASS | Valid repair |
| deep_get | FAIL | PASS | Valid repair |
| tree_depth_tuple | FAIL | PASS | Spec-conflicted repair |

## Generalisation benchmark (v2.10) — 32 tasks, 5 families

| Family | Champion | Repair | Delta |
|---|---|---|---|
| interval_merge (6) | 4/6 = 66.7% | 5/6 = 83.3% | +16.7 pp |
| sorted_selection (7) | 3/7 = 42.9% | 3/7 = 42.9% | 0 pp |
| nested_dict (7) | 6/7 = 85.7% | 5/7 = 71.4% | −14.3 pp |
| tuple_tree (7) | 3/7 = 42.9% | 2/7 = 28.6% | −14.3 pp |
| rle_encoding (5) | 4/5 = 80.0% | 3/5 = 60.0% | −20.0 pp |
| **Total (32)** | **20/32 = 62.5%** | **18/32 = 56.2%** | **−6.3 pp** |

## Routing audit (v2.11) — same 32 tasks

| Router | Score | vs Champion |
|---|---|---|
| Family router | 19/32 = 59.4% | −3.1 pp |
| Confidence router | 20/32 = 62.5% | 0 pp |
| Oracle ceiling | 23/32 = 71.9% | +9.4 pp |

**Oracle analysis:** 15 tasks both-pass, 5 champion-only, 3 repair-only, 9 both-fail.
Maximum recoverable via routing = 3 tasks (the repair-only group).

## Score trends

| Benchmark | Adapter only | + Memory | + Repair (diag.) |
|---|---|---|---|
| Frozen 28-task | 64.3% | 82.1% | 96.4%* |
| 32 clean tasks | — | 62.5% | 56.2% |

*Diagnostic only; frozen benchmark is no longer independent after repair targeting.
