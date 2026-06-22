# v2.18 Per-Family Breakdown

**Status:** PENDING — populate after Phase A + B runs complete.

## Families (32-task clean benchmark)

| Family | n | TF-IDF run1 | TF-IDF run2 | TF-IDF run3 | TF-IDF stable | Code-dense | Code-hybrid |
|---|---|---|---|---|---|---|---|
| interval (6) | 6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| sorting/heap (6) | 6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| nested_dict (6) | 6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| tuple_tree (6) | 6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| rle_string (5) | 5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| **Total** | **32** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Flip hypotheses

Based on v2.17 MiniLM results, these tasks showed per-run variability:
- `meeting_rooms` — flipped across MiniLM and TF-IDF
- `range_summary` — flipped across MiniLM and TF-IDF
- `running_median` — flipped across MiniLM and TF-IDF
- `unflatten_dict` — flipped across MiniLM and TF-IDF
- `find_peak_element` — flipped across MiniLM and TF-IDF

Tasks consistently failing across all modes (likely model capability ceiling):
- `interval_union`, `insert_interval`, `kth_smallest_matrix`, `wiggle_sort`
- `count_smaller_after_self`, `deep_delete`, `tree_max_path_sum`, `tree_from_list`
- `tree_serialize`, `tree_width`, `rle_delta_encode`
