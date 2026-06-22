# Benchmark Registry

This document registers all evaluation benchmarks used in the AetherForge v2.6–v2.13
research arc. Each benchmark is classified by contamination status (clean, diagnostic,
or oracle-only).

---

## Benchmark 1 — Frozen 28-Task Benchmark

**Status:** CLEAN
**Tasks:** 28
**File:** `data/heldout_code_agent_tasks.jsonl` (and equivalents in eval task files)
**Used in:** v2.7 champion audit, v2.8 top-k/prompt audit, v2.9 repair diagnostic

### Description

Hard single-function Python coding tasks. Each task contains:
- A natural language description of the function to implement
- Expected properties verified by test assertions
- Execution confirmation (PASS requires assertion to hold)

Categories include graph traversal, caching, string processing, sorting, nested data
structures, set operations, and more.

### Contamination status

| Evaluation | Status | Reason |
|---|---|---|
| Champion index (23/28) | **Clean** | Index built before benchmark was finalized; no task overlap |
| Repair index diagnostic (27/28) | **Diagnostic only** | 4 repair records target known frozen-benchmark failures |
| Any run after repair index was built | Diagnostic | Benchmark is no longer independent for the repair index |

### Known defect

Task `tree_depth_tuple` contains a spec-conflicted assertion. The prompt expects
`tree_depth(((1,2),(3,(4,5)))) == 3`, but by the stated recursive rule the correct
value is 4. A correct implementation fails this assertion. This is reported
transparently; all results are given as both raw and corrected-audit scores.

---

## Benchmark 2 — v2.10 32-Task Clean Generalisation Benchmark

**Status:** CLEAN (zero overlap with frozen 28-task benchmark and v2.9 5-task set)
**Tasks:** 32
**File:** `data/v210_clean_repair_generalisation_tasks.jsonl`
**Used in:** v2.10 global repair audit, v2.11 routing audit

### Description

32 tasks across 5 algorithm families:

| Family | Tasks | Description |
|---|---|---|
| interval_merge | 6 | Interval merging, scheduling, union |
| sorted_selection | 7 | Sorted arrays, median, kth element |
| nested_dict | 7 | Nested dictionary access and update |
| tuple_tree | 7 | Recursive tuple-tree operations |
| rle_encoding | 5 | Run-length encoding/decoding |

Constructed with zero overlap guarantee: none of the 32 task IDs or prompts appear in
the frozen 28-task benchmark or the v2.9 5-task repair-generalisation set.

### Results

| Index | Score |
|---|---|
| Champion (memory/index_adapted) | 20/32 = 62.5% |
| Repair (memory/index_adapted_v29) | 18/32 = 56.2% |

Champion is the stronger configuration on clean generalisation tasks.

---

## Benchmark 3 — v2.9 5-Task Repair-Generalisation Set

**Status:** CLEAN (small)
**Tasks:** 5
**Used in:** v2.9 early transfer signal

### Description

Five tasks similar in pattern to the 4 repair targets (merge, median, nested-dict,
tree-depth), but not identical. Used to test whether repair memory transfers to
nearby tasks before committing to the 32-task v2.10 benchmark.

**Result:** 4/5 = 80% with repair index. Positive signal, but too small for a strong
claim. Motivated the larger v2.10 test.

---

## Benchmark 4 — Oracle Routing Ceiling (v2.11, Diagnostic)

**Status:** Diagnostic / oracle-only
**Tasks:** 32 (same as Benchmark 2)
**Used in:** v2.11 routing audit

### Description

Per-task selection of whichever index (champion or repair) achieved PASS in v2.10.
Not a deployable routing strategy — requires knowing the answer in advance.

**Result:** 23/32 = 71.9%
- 15 tasks: both indexes pass
- 5 tasks: champion only
- 3 tasks: repair only (oracle-recoverable)
- 9 tasks: both fail (unreachable by any index selection)

---

## Benchmark Design Rules

1. No task may appear in both a training/repair set and an evaluation benchmark.
2. Clean benchmarks must be held out before any experiment using them.
3. If a benchmark becomes non-independent (e.g., repair records target known failures),
   all results on that benchmark must be labelled diagnostic.
4. Spec-conflicted tasks must be documented and reported transparently.
5. All clean benchmark results use best-of-3 sampling with verified_agent scoring and
   strict agent contract.
