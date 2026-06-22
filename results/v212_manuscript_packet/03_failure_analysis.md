# AetherForge — Failure Analysis

## Structural failures (both indexes fail on frozen benchmark)

5 tasks fail regardless of memory configuration. These are not retrieval failures;
they represent model capability limits or task difficulty that memory cannot address.

| Task | Category | Failure pattern |
|---|---|---|
| merge_intervals | hard | Retrieval noise: 8 `merge_sorted` duplicates dominate k=4 retrieval |
| median_two_sorted | hard | Structural gap: no close pattern in memory; model never generates correct 2-pointer merge |
| deep_get | hard | Structural gap: invert_dict and flatten records dominate; wrong key-iteration pattern injected |
| tree_depth_tuple | hard | Spec conflict: task prompt asserts wrong value (==3); correct is ==4; + retrieval noise from flatten |
| rle_decode | hard | Consistent failure; no pattern overlap in champion memory |

(After v2.9 repair: all 4 repair targets were fixed diagnostically. Only `rle_decode` remains a
structural failure in the champion-index configuration.)

## Retrieval noise taxonomy

Three distinct mechanisms cause harmful retrieval in the champion index:

### Type 1 — Surface lexical overlap (vocabulary collision)
**Affected tasks:** `merge_intervals`, `median_two_sorted`

The TF-IDF embedder assigns high similarity to records sharing high-frequency tokens
("merge", "two", "sorted") regardless of algorithmic intent. The champion index contains
8 `merge_sorted` records that dominate k=4 retrieval for both tasks.

Top-1 retrieval scores for merge_intervals in champion index:
- `merge_sorted` record: 0.5572 (HARMFUL — wrong pattern)
- Expected `merge_intervals` pattern: absent from index

### Type 2 — Structural vocabulary overlap
**Affected tasks:** `tree_depth_tuple`, `deep_get`

"Nested" vocabulary (flatten, dict, nested, depth) causes records from unrelated patterns
to rank highly. `flatten` records dominate `tree_depth_tuple` retrieval (0.3902);
`invert_dict` + `flatten` records dominate `deep_get` (0.3958).

### Type 3 — Repair vocabulary leak (v2.10 / v2.11 finding)
**Affected tasks:** All nested_dict and tuple_tree tasks when repair index is used

After adding 4 repair records, the `deep_get` repair record scores 0.64–0.82
against ALL 7 nested_dict tasks. The `tree_depth` repair record scores 0.59–0.72
against ALL 7 tuple_tree tasks. This is because the repair records are densely
written with the full algorithmic vocabulary of their family.

Result: repair routing fires on entire families, not specific failing tasks.

## Benchmark defect (v2.9 finding)

`tree_depth_tuple` task prompt asserts `tree_depth(((1,2),(3,(4,5)))) == 3`.
By the stated rule (leaf = depth 1, branch = 1 + max(left, right)):

```
tree_depth((1,2))           = 1 + max(1,1) = 2
tree_depth((3,(4,5)))       = 1 + max(1,2) = 3
tree_depth(((1,2),(3,(4,5)))) = 1 + max(2,3) = 4
```

Correct value is **4**, not 3. The assertion in the task is wrong.
Any correct implementation returns 4; the task FAIL with champion memory means the
model either copies the broken assertion verbatim or otherwise fails.
After repair, the PASS means the model follows repair memory's correct ==4 assertion.

**Action taken:** Documented as spec defect. Dual-score reporting used in all v2.9
summaries (raw 26/28 and corrected 25/27). Not silently removed from benchmark.

## Sampling variance characterisation

Across all eval runs (v2.10, v2.11 family/confidence sub-evals), individual tasks
show outcome variance:

| Task | Run A | Run B |
|---|---|---|
| interval_union | PASS (repair) | FAIL (repair) |
| insert_interval | FAIL (repair) | PASS (repair) |
| merge_sorted_k | PASS (champion) | FAIL (champion) |
| deep_delete | PASS (champion) | FAIL (champion) |
| rle_expand | PASS (champion) | FAIL (champion) |

At best-of-3 sampling, variance is approximately ±2–3 tasks on the 32-task benchmark.
Any routing gain of fewer than 3 tasks cannot be distinguished from noise.

## Root cause of all retraining failures (v2.5 / v2.6)

**Common cause:** 2e-5 learning rate + new data mixture overwrites the generalisation
properties established by the original 300-step, 6e-6 LR, agent-only training.

Symptoms:
- Hard tasks: +24 pp improvement (traces add step-by-step reasoning)
- String tasks: −50 pp regression
- Basic tasks: −100 pp regression (all fail)

The original adapter's success on string and basic tasks depends on training
trajectory properties that cannot be restored through merge-and-retrain at any
trace ratio tested. Five independent rejections confirm this.

## Tasks unsolvable by either index (v2.10 / v2.11)

9 of 32 clean tasks fail with both champion and repair indexes:

```
v210_kth_smallest_matrix
v210_wiggle_sort
v210_count_smaller_after_self
v210_tree_max_path_sum
v210_tree_from_list
v210_tree_serialize
v210_tree_width
v210_rle_delta_encode
v210_insert_interval (some runs)
```

These tasks require either more complex reasoning (tree width via BFS, kth in matrix),
algorithms not covered by current memory (wiggle sort, delta encoding), or
multi-step implementations the model cannot generate in 3 best-of-n attempts.
They are not retrieval failures — they require model improvement.
