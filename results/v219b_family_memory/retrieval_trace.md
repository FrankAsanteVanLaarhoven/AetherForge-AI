# v2.19b Retrieval Trace — Changed Tasks

Top structured-dense retrievals over the COMBINED pool for tasks whose pass count
changed vs baseline. [NEW] marks the v2.19b family-targeted records. Contrast with
v2.19, where interval/tree/rle tasks surfaced family-irrelevant records.

### `v210_deep_delete`  (family dict, baseline 0/3 → v219b 1/3)
- [NEW][✓ dict] `deep_get(d, path, default=None)` score=0.900
- [NEW][✓ dict] `deep_values(d)` score=0.874
- [NEW][✓ dict] `deep_max_depth(d)` score=0.858
- [   ][✓ dict] `invert_dict(d)` score=0.789

### `v210_deep_filter`  (family dict, baseline 2/3 → v219b 3/3)
- [NEW][✓ dict] `deep_values(d)` score=0.785
- [NEW][✓ dict] `deep_get(d, path, default=None)` score=0.736
- [NEW][✓ dict] `deep_max_depth(d)` score=0.700
- [   ][✓ dict] `invert_dict(d)` score=0.661

### `v210_deep_keys`  (family dict, baseline 1/3 → v219b 3/3)
- [NEW][✓ dict] `deep_values(d)` score=0.918
- [NEW][✓ dict] `deep_get(d, path, default=None)` score=0.837
- [NEW][✓ dict] `deep_max_depth(d)` score=0.724
- [   ][✓ dict] `flatten(lst)` score=0.694

### `v210_deep_update_if`  (family dict, baseline 3/3 → v219b 2/3)
- [NEW][✓ dict] `deep_values(d)` score=0.928
- [NEW][✓ dict] `deep_max_depth(d)` score=0.765
- [NEW][✓ dict] `deep_get(d, path, default=None)` score=0.756
- [   ][✓ dict] `invert_dict(d)` score=0.706

### `v210_find_peak_element`  (family search, baseline 1/3 → v219b 3/3)
- [   ][✓ search] `binary_search(arr, target)` score=0.713
- [   ][✓ search] `binary_search(arr, target)` score=0.713
- [   ][✓ search] `binary_search(arr, target)` score=0.713
- [   ][✓ search] `binary_search(arr, target)` score=0.713

### `v210_insert_interval`  (family interval, baseline 1/3 → v219b 0/3)
- [NEW][✓ interval] `can_attend_meetings(intervals)` score=0.877
- [NEW][✓ interval] `total_covered_length(intervals)` score=0.829
- [NEW][✓ interval] `point_coverage_count(intervals, point)` score=0.790
- [NEW][✓ interval] `min_arrows(points)` score=0.773

### `v210_interval_intersection`  (family interval, baseline 0/3 → v219b 3/3)
- [NEW][✓ interval] `can_attend_meetings(intervals)` score=0.878
- [NEW][✓ interval] `total_covered_length(intervals)` score=0.802
- [NEW][✓ interval] `min_arrows(points)` score=0.788
- [NEW][✓ interval] `point_coverage_count(intervals, point)` score=0.766

### `v210_meeting_rooms`  (family interval, baseline 2/3 → v219b 3/3)
- [NEW][✓ interval] `can_attend_meetings(intervals)` score=0.924
- [NEW][✓ interval] `min_arrows(points)` score=0.787
- [NEW][✓ interval] `total_covered_length(intervals)` score=0.739
- [NEW][✓ interval] `point_coverage_count(intervals, point)` score=0.738

### `v210_range_summary`  (family interval, baseline 1/3 → v219b 3/3)
- [NEW][✓ interval] `min_arrows(points)` score=0.720
- [NEW][✓ interval] `total_covered_length(intervals)` score=0.718
- [NEW][✓ interval] `can_attend_meetings(intervals)` score=0.715
- [NEW][✓ interval] `point_coverage_count(intervals, point)` score=0.697

### `v210_rle_compress`  (family rle, baseline 2/3 → v219b 3/3)
- [NEW][✓ rle] `rle_most_common(pairs)` score=0.742
- [NEW][✓ rle] `rle_distinct_chars(pairs)` score=0.697
- [NEW][✓ rle] `rle_total_length(pairs)` score=0.645
- [NEW][✓ rle] `rle_char_at(pairs, idx)` score=0.628

### `v210_tree_leaves`  (family tree, baseline 3/3 → v219b 2/3)
- [NEW][✓ tree] `tree_contains(node, target)` score=0.988
- [NEW][✓ tree] `tree_height(node)` score=0.959
- [NEW][✓ tree] `tree_count_nodes(node)` score=0.944
- [NEW][✓ tree] `tree_min_leaf(node)` score=0.936

### `v210_unflatten_dict`  (family dict, baseline 2/3 → v219b 1/3)
- [NEW][✓ dict] `deep_values(d)` score=0.806
- [   ][✓ dict] `flatten(lst)` score=0.796
- [   ][✓ dict] `flatten(lst)` score=0.795
- [   ][✓ dict] `flatten(lst)` score=0.795

