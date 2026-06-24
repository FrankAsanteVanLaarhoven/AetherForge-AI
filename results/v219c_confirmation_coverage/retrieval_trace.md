# v2.19c Retrieval Trace — Persistent-Failure & Changed Tasks

Top expanded-pool retrievals for the persistent v2.19b failures and any task whose expanded pass count changed. [v219c]/[v219b] mark added records.

### `v210_deep_keys`  (dict, baseline 1/3 → expanded 3/3)
- [v219b][✓ dict] `deep_values(d)` score=0.918
- [v219b][✓ dict] `deep_get(d, path, default=None)` score=0.837
- [v219b][✓ dict] `deep_max_depth(d)` score=0.724
- [orig ][✓ dict] `flatten(lst)` score=0.694

### `v210_deep_update_if`  (dict, baseline 3/3 → expanded 2/3)
- [v219b][✓ dict] `deep_values(d)` score=0.928
- [v219b][✓ dict] `deep_max_depth(d)` score=0.765
- [v219b][✓ dict] `deep_get(d, path, default=None)` score=0.756
- [orig ][✓ dict] `invert_dict(d)` score=0.706

### `v210_insert_interval`  (interval, baseline 1/3 → expanded 0/3)
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.978
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.944
- [v219b][✓ interval] `can_attend_meetings(intervals)` score=0.877
- [v219c][✓ interval] `interval_clip(intervals, lo, hi)` score=0.837

### `v210_interval_intersection`  (interval, baseline 0/3 → expanded 3/3)
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.924
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.918
- [v219b][✓ interval] `can_attend_meetings(intervals)` score=0.878
- [v219c][✓ interval] `interval_clip(intervals, lo, hi)` score=0.817

### `v210_interval_union`  (interval, baseline 0/3 → expanded 3/3)
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.946
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.931
- [v219b][✓ interval] `can_attend_meetings(intervals)` score=0.860
- [v219b][✓ interval] `total_covered_length(intervals)` score=0.830

### `v210_meeting_rooms`  (interval, baseline 2/3 → expanded 3/3)
- [v219b][✓ interval] `can_attend_meetings(intervals)` score=0.924
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.831
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.804
- [v219b][✓ interval] `min_arrows(points)` score=0.787

### `v210_non_overlapping_intervals`  (interval, baseline 3/3 → expanded 2/3)
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.904
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.871
- [v219b][✓ interval] `min_arrows(points)` score=0.834
- [v219b][✓ interval] `can_attend_meetings(intervals)` score=0.821

### `v210_range_summary`  (interval, baseline 1/3 → expanded 3/3)
- [v219c][✓ interval] `interval_gaps(intervals)` score=0.850
- [v219c][✓ interval] `interval_complement(intervals, lo, hi)` score=0.801
- [v219c][✓ interval] `interval_clip(intervals, lo, hi)` score=0.721
- [v219b][✓ interval] `min_arrows(points)` score=0.720

### `v210_rle_expand`  (rle, baseline 2/3 → expanded 3/3)
- [v219b][✓ rle] `rle_total_length(pairs)` score=0.755
- [v219b][✓ rle] `rle_char_at(pairs, idx)` score=0.724
- [v219b][✓ rle] `rle_distinct_chars(pairs)` score=0.698
- [v219b][✓ rle] `rle_most_common(pairs)` score=0.698

### `v210_rle_roundtrip`  (rle, baseline 3/3 → expanded 1/3)
- [orig ][✓ rle] `rle_encode(s)` score=0.867
- [v219b][✓ rle] `rle_char_at(pairs, idx)` score=0.843
- [v219b][✓ rle] `rle_total_length(pairs)` score=0.827
- [v219b][✓ rle] `rle_most_common(pairs)` score=0.814

### `v210_running_median`  (search, baseline 2/3 → expanded 1/3)
- [orig ][✓ search] `binary_search(arr: List[int], target: int)` score=0.621
- [orig ][✓ search] `binary_search(arr: List[int], target: int)` score=0.621
- [orig ][✓ search] `binary_search(arr: List[int], target: int)` score=0.620
- [orig ][✓ search] `binary_search(arr, target)` score=0.598

### `v210_tree_from_list`  (tree, baseline 0/3 → expanded 0/3)
- [v219c][✓ tree] `tree_level_counts(node)` score=0.890
- [v219b][✓ tree] `tree_height(node)` score=0.877
- [v219b][✓ tree] `tree_count_nodes(node)` score=0.861
- [v219c][✓ tree] `tree_count_at_depth(node, depth)` score=0.851

### `v210_tree_max_path_sum`  (tree, baseline 0/3 → expanded 0/3)
- [v219b][✓ tree] `tree_contains(node, target)` score=0.993
- [v219b][✓ tree] `tree_height(node)` score=0.987
- [v219b][✓ tree] `tree_min_leaf(node)` score=0.964
- [v219c][✓ tree] `tree_level_counts(node)` score=0.951

### `v210_tree_serialize`  (tree, baseline 0/3 → expanded 0/3)
- [v219c][✓ tree] `tree_to_nested_list(node)` score=0.907
- [v219b][✓ tree] `tree_contains(node, target)` score=0.888
- [v219b][✓ tree] `tree_height(node)` score=0.869
- [v219b][✓ tree] `tree_count_nodes(node)` score=0.869

### `v210_tree_width`  (tree, baseline 0/3 → expanded 0/3)
- [v219c][✓ tree] `tree_count_at_depth(node, depth)` score=1.026
- [v219c][✓ tree] `tree_level_counts(node)` score=1.016
- [v219b][✓ tree] `tree_height(node)` score=1.013
- [v219b][✓ tree] `tree_count_nodes(node)` score=0.972

