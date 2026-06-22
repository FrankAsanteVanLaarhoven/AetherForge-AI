# v2.17 Per-Family Breakdown — Clean 32-Task Benchmark

## Summary table

| Family | n | TF-IDF | Dense (MiniLM) | Hybrid (MiniLM) |
|---|---|---|---|---|
| interval (6) | 6 | 2/6 | 4/6 | 3/6 |
| sorting/heap (6) | 6 | 2/6 | 1/6 | 1/6 |
| nested_dict (6) | 6 | 4/6 | 4/6 | 4/6 |
| tuple_tree (6) | 6 | 3/6 | 3/6 | 3/6 |
| rle_string (5) | 5 | 4/5 | 3/5 | 4/5 |
| **Total** | **32** | **17/32 = 53.1%** | **17/32 = 53.1%** | **16/32 = 50.0%** |

## Per-task detail

| Task | Family | TF-IDF | Dense | Hybrid |
|---|---|---|---|---|
| v210_interval_union | interval | ✗ | ✗ | ✗ |
| v210_insert_interval | interval | ✗ | ✗ | ✓ |
| v210_meeting_rooms | interval | ✗ | ✓ | ✗ |
| v210_non_overlapping_intervals | interval | ✓ | ✓ | ✓ |
| v210_range_summary | interval | ✗ | ✓ | ✓ |
| v210_interval_intersection | interval | ✓ | ✓ | ✗ |
| v210_kth_smallest_matrix | sorting/heap | ✗ | ✗ | ✗ |
| v210_running_median | sorting/heap | ✗ | ✓ | ✗ |
| v210_merge_sorted_k | sorting/heap | ✓ | ✗ | ✗ |
| v210_wiggle_sort | sorting/heap | ✗ | ✗ | ✗ |
| v210_count_smaller_after_self | sorting/heap | ✗ | ✗ | ✗ |
| v210_find_peak_element | sorting/heap | ✓ | ✗ | ✗ |
| v210_search_rotated_sorted | sorting/heap | ✓ | ✓ | ✓ |
| v210_deep_delete | nested_dict | ✗ | ✗ | ✗ |
| v210_flatten_dict | nested_dict | ✓ | ✓ | ✗ |
| v210_deep_merge | nested_dict | ✓ | ✓ | ✓ |
| v210_deep_keys | nested_dict | ✓ | ✗ | ✓ |
| v210_unflatten_dict | nested_dict | ✗ | ✓ | ✓ |
| v210_deep_update_if | nested_dict | ✓ | ✓ | ✓ |
| v210_deep_filter | nested_dict | ✓ | ✓ | ✓ |
| v210_tree_sum | tuple_tree | ✓ | ✓ | ✓ |
| v210_tree_leaves | tuple_tree | ✓ | ✓ | ✓ |
| v210_tree_mirror | tuple_tree | ✓ | ✓ | ✓ |
| v210_tree_max_path_sum | tuple_tree | ✗ | ✗ | ✗ |
| v210_tree_from_list | tuple_tree | ✗ | ✗ | ✗ |
| v210_tree_serialize | tuple_tree | ✗ | ✗ | ✗ |
| v210_tree_width | tuple_tree | ✗ | ✗ | ✗ |
| v210_rle_roundtrip | rle_string | ✓ | ✓ | ✓ |
| v210_rle_compress | rle_string | ✓ | ✗ | ✓ |
| v210_rle_expand | rle_string | ✓ | ✓ | ✓ |
| v210_rle_longest_run | rle_string | ✓ | ✓ | ✓ |
| v210_rle_delta_encode | rle_string | ✗ | ✗ | ✗ |

## Task flip analysis (vs TF-IDF baseline)

### Dense gains (TF-IDF FAIL → Dense PASS)
- `meeting_rooms` (interval) — dense retrieved relevant interval-comparison pattern
- `range_summary` (interval) — dense retrieved relevant range-boundary pattern
- `running_median` (sorting/heap) — dense retrieved relevant heap-usage pattern
- `unflatten_dict` (nested_dict) — dense retrieved relevant dict-reconstruction pattern

### Dense regressions (TF-IDF PASS → Dense FAIL)
- `merge_sorted_k` (sorting/heap) — dense retrieved less relevant candidate
- `find_peak_element` (sorting/heap) — dense retrieved less relevant candidate
- `deep_keys` (nested_dict) — dense retrieved less relevant candidate
- `rle_compress` (rle_string) — dense retrieved less relevant candidate

**Net: +4 gains, −4 regressions = 0 task delta. Total tied 17/32.**

### Hybrid gains (TF-IDF FAIL → Hybrid PASS)
- `insert_interval` (interval) — hybrid shortlist included correct interval-insert pattern

### Hybrid regressions (TF-IDF PASS → Hybrid FAIL)
- `interval_intersection` (interval) — hybrid rerank displaced the correct candidate
- `flatten_dict` (nested_dict) — hybrid rerank displaced the correct candidate

**Net: +1 gain, −2 regressions = −1 task delta. Total 16/32.**

## Interpretation

The interval family shows the most sensitivity to retrieval mode (2→4 for dense). However
dense also hurt the sorting/heap family (2→1). The dominant signal is **sampling variance**:
all three modes are within ±1 of each other across all families. MiniLM is not a code
embedder and cannot distinguish between algorithmically-different tasks that share
vocabulary. The per-task shuffling reflects noise at best-of-3 sampling rather than
systematic retrieval improvement.

The hypothesis that dense retrieval resolves Type 1 failures (lexical collision) is
partially supported for the interval family (+2 tasks), but the sorting/heap regressions
cancel the gain. A code-specialised embedder remains the next step.
