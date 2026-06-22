# v2.19 Retrieval Trace — Changed Tasks

Top structured-dense retrievals (deterministic; re-run offline) for tasks whose
pass count changed vs the baseline. Shows which memory records the structured
multi-view reranker surfaces — exposing whether the memory pool actually contains
family-relevant repair signal for these tasks.

### `v210_deep_filter`  (query family: dict, baseline 2/3 → structured_dense 3/3, structured_hybrid 2/3)
- [✓ family=dict] `invert_dict(d)` score=0.661 id=2dab6852
- [✓ family=dict] `flatten(lst)` score=0.652 id=0e8acb07
- [✓ family=dict] `flatten(lst)` score=0.651 id=a98014cb
- [✓ family=dict] `flatten(lst)` score=0.651 id=76a358a3

### `v210_deep_keys`  (query family: dict, baseline 1/3 → structured_dense 2/3, structured_hybrid 3/3)
- [✓ family=dict] `flatten(lst)` score=0.694 id=0e8acb07
- [✓ family=dict] `flatten(lst)` score=0.693 id=a98014cb
- [✓ family=dict] `flatten(lst)` score=0.693 id=76a358a3
- [✓ family=dict] `flatten(lst)` score=0.692 id=fafef2cf

### `v210_deep_update_if`  (query family: dict, baseline 3/3 → structured_dense 3/3, structured_hybrid 1/3)
- [✓ family=dict] `invert_dict(d)` score=0.706 id=2dab6852
- [✓ family=dict] `flatten(lst)` score=0.692 id=0e8acb07
- [✓ family=dict] `flatten(lst)` score=0.691 id=a98014cb
- [✓ family=dict] `flatten(lst)` score=0.691 id=76a358a3

### `v210_find_peak_element`  (query family: search, baseline 1/3 → structured_dense 2/3, structured_hybrid 2/3)
- [✓ family=search] `binary_search(arr, target)` score=0.713 id=29edffed
- [✓ family=search] `binary_search(arr, target)` score=0.713 id=6a4a1de0
- [✓ family=search] `binary_search(arr, target)` score=0.713 id=a318d781
- [✓ family=search] `binary_search(arr, target)` score=0.713 id=d2fd6c50

### `v210_flatten_dict`  (query family: dict, baseline 3/3 → structured_dense 2/3, structured_hybrid 2/3)
- [✓ family=dict] `flatten(lst)` score=0.986 id=0e8acb07
- [✓ family=dict] `flatten(lst)` score=0.985 id=a98014cb
- [✓ family=dict] `flatten(lst)` score=0.985 id=76a358a3
- [✓ family=dict] `flatten(lst)` score=0.984 id=fafef2cf

### `v210_insert_interval`  (query family: interval, baseline 1/3 → structured_dense 0/3, structured_hybrid 0/3)
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.575 id=d42169fe
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.575 id=c6f277d5
- [✗ family=sort] `merge_sorted(a, b)` score=0.549 id=c62b0b69
- [✗ family=sort] `merge_sorted(a, b)` score=0.524 id=2eac1a8d

### `v210_interval_intersection`  (query family: interval, baseline 0/3 → structured_dense 2/3, structured_hybrid 2/3)
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.566 id=d42169fe
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.565 id=c6f277d5
- [✗ family=sort] `merge_sorted(a, b)` score=0.556 id=2eac1a8d
- [✗ family=sort] `merge_sorted(a, b)` score=0.556 id=33312ca6

### `v210_interval_union`  (query family: interval, baseline 0/3 → structured_dense 0/3, structured_hybrid 1/3)
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.691 id=d42169fe
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.691 id=c6f277d5
- [✗ family=sort] `merge_sorted(a, b)` score=0.685 id=2eac1a8d
- [✗ family=sort] `merge_sorted(a, b)` score=0.685 id=33312ca6

### `v210_kth_smallest_matrix`  (query family: matrix, baseline 0/3 → structured_dense 3/3, structured_hybrid 0/3)
- [✓ family=matrix] `transpose(matrix)` score=0.637 id=b758c0bf
- [✗ family=sort] `unique_sorted(lst)` score=0.465 id=f6a82135
- [✗ family=sort] `unique_sorted(lst)` score=0.464 id=f688f802
- [✗ family=sort] `unique_sorted(lst)` score=0.464 id=660a85db

### `v210_meeting_rooms`  (query family: interval, baseline 2/3 → structured_dense 3/3, structured_hybrid 3/3)
- [✗ family=misc] `two_sum(nums: List[int], target: int)` score=0.415 id=1b58e337
- [✗ family=string] `is_anagram(s, t)` score=0.385 id=f5a71946
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.384 id=d42169fe
- [✗ family=sort] `merge_sorted(a: List[int], b: List[int])` score=0.384 id=c6f277d5

### `v210_range_summary`  (query family: interval, baseline 1/3 → structured_dense 3/3, structured_hybrid 3/3)
- [✗ family=sort] `unique_sorted(lst: List[int])` score=0.565 id=606c955f
- [✗ family=sort] `unique_sorted(lst)` score=0.533 id=f6a82135
- [✗ family=sort] `unique_sorted(lst)` score=0.532 id=f688f802
- [✗ family=sort] `unique_sorted(lst)` score=0.532 id=660a85db

### `v210_rle_compress`  (query family: rle, baseline 2/3 → structured_dense 1/3, structured_hybrid 2/3)
- [✓ family=rle] `rle_encode(s)` score=0.594 id=50461b39
- [✗ family=dict] `flatten(lst)` score=0.553 id=0e8acb07
- [✗ family=dict] `flatten(lst)` score=0.552 id=a98014cb
- [✗ family=dict] `flatten(lst)` score=0.552 id=76a358a3

### `v210_rle_delta_encode`  (query family: rle, baseline 0/3 → structured_dense 0/3, structured_hybrid 1/3)
- [✓ family=rle] `rle_encode(s)` score=0.721 id=50461b39
- [✗ family=dict] `flatten(lst: List[List[int]])` score=0.517 id=eafbbf21
- [✗ family=dict] `flatten(lst)` score=0.510 id=0e8acb07
- [✗ family=dict] `flatten(lst)` score=0.509 id=a98014cb

### `v210_rle_expand`  (query family: rle, baseline 2/3 → structured_dense 3/3, structured_hybrid 3/3)
- [✗ family=dict] `flatten(lst)` score=0.644 id=0e8acb07
- [✗ family=dict] `flatten(lst)` score=0.643 id=a98014cb
- [✗ family=dict] `flatten(lst)` score=0.643 id=76a358a3
- [✗ family=dict] `flatten(lst)` score=0.642 id=fafef2cf

### `v210_rle_roundtrip`  (query family: rle, baseline 3/3 → structured_dense 2/3, structured_hybrid 2/3)
- [✓ family=rle] `rle_encode(s)` score=0.867 id=50461b39
- [✗ family=string] `is_anagram(s, t)` score=0.528 id=f5a71946
- [✗ family=string] `word_count(text)` score=0.505 id=6483a017
- [✗ family=string] `is_palindrome(s)` score=0.486 id=5728af2c

### `v210_running_median`  (query family: search, baseline 2/3 → structured_dense 0/3, structured_hybrid 2/3)
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.621 id=fc8a10c0
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.621 id=86e611da
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.620 id=3fb90e6b
- [✓ family=search] `binary_search(arr, target)` score=0.598 id=29edffed

### `v210_tree_width`  (query family: tree, baseline 0/3 → structured_dense 0/3, structured_hybrid 1/3)
- [✗ family=graph] `bfs(graph, start)` score=0.483 id=4432511a
- [✗ family=graph] `bfs(graph, start)` score=0.483 id=fd119198
- [✗ family=dict] `flatten(lst: List[List[int]])` score=0.451 id=eafbbf21
- [✗ family=graph] `bfs(graph, start)` score=0.451 id=5e2ecf39

### `v210_unflatten_dict`  (query family: dict, baseline 2/3 → structured_dense 1/3, structured_hybrid 3/3)
- [✓ family=dict] `flatten(lst)` score=0.796 id=0e8acb07
- [✓ family=dict] `flatten(lst)` score=0.795 id=a98014cb
- [✓ family=dict] `flatten(lst)` score=0.795 id=76a358a3
- [✓ family=dict] `flatten(lst)` score=0.794 id=fafef2cf

### `v210_wiggle_sort`  (query family: search, baseline 0/3 → structured_dense 0/3, structured_hybrid 1/3)
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.647 id=86e611da
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.646 id=3fb90e6b
- [✓ family=search] `binary_search(arr: List[int], target: int)` score=0.622 id=fc8a10c0
- [✗ family=sort] `unique_sorted(lst)` score=0.609 id=f6a82135

