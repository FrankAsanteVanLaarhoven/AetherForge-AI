# Figure Spec: Retrieval Failure Taxonomy

**Figure type:** Three-column taxonomy diagram (table or flow diagram)
**Caption:** Three classes of retrieval failure observed in AetherForge. Each class shows the failing task, the retrieved record, the overlap mechanism, and the effect on agent output.

## Columns

### Column 1 — Surface Lexical Collision

```
Failing task:     merge_intervals
Retrieved record: merge_sorted (×8 in index)
Overlap token:    "merge", "sorted"
Similarity score: 0.5572 (top-1)
Effect:           Agent generates merge_sorted pattern; output fails interval-merge assertion
```

### Column 2 — Structural Vocabulary Overlap

```
Failing task:     tree_depth_tuple
Retrieved record: flatten (×8 in index)
Overlap token:    "nested", "list", "depth"
Similarity score: 0.3902 (top-1)
Effect:           Agent generates recursive flatten pattern; returns flattened list not integer depth

Failing task:     deep_get
Retrieved record: invert_dict, flatten
Overlap token:    "dict", "nested", "key"
Similarity score: 0.3958, 0.3656 (top-2)
Effect:           Agent generates dict inversion instead of nested key traversal
```

### Column 3 — Repair Vocabulary Leak (v2.10 / v2.11 finding)

```
Repair record:    deep_get (fix for nested dict access)
Fires on:         ALL 7 nested_dict tasks (similarity 0.64–0.82)
Intended target:  1 task (deep_get pattern)
Effect:           Routing heuristics fire on entire family; tasks that pass with champion
                  (deep_delete, unflatten_dict) regress when repair record is retrieved

Repair record:    tree_depth_tuple
Fires on:         ALL 7 tuple_tree tasks (similarity 0.59–0.72)
Intended target:  1 task (recursive depth pattern)
Effect:           Same mechanism: tree_mirror, tree_leaves pass with champion but can
                  regress when tree_depth repair record is retrieved
```

## Key takeaway for figure

> TF-IDF similarity cannot distinguish "this task requires this repair pattern" from
> "this task shares vocabulary with this repair family." All three failure classes arise
> from the same root cause: bag-of-words similarity conflates algorithm identity with
> lexical co-occurrence.

## Suggested layout

Three boxes in a row, each showing: task, retrieved record, overlap tokens (highlighted), effect arrow pointing to "wrong output" or "correct output".
A root-cause label below all three: "Root cause: TF-IDF similarity ≠ algorithm similarity."
