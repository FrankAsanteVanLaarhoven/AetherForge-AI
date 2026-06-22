# v2.17 Per-Family Breakdown

**Status:** PENDING — populate after running eval targets.

## Families (clean 32-task benchmark)

| Family | n_tasks | tfidf_pass | dense_pass | hybrid_pass |
|---|---|---|---|---|
| string_processing | 6 | PENDING | PENDING | PENDING |
| list_algorithms | 6 | PENDING | PENDING | PENDING |
| nested_dict | 7 | PENDING | PENDING | PENDING |
| tuple_tree | 7 | PENDING | PENDING | PENDING |
| interval | 3 | PENDING | PENDING | PENDING |
| sorting | 3 | PENDING | PENDING | PENDING |

## Key Hypotheses to Test

1. **Type 1 failures (surface lexical collision):** `merge_intervals` / `merge_sorted`.
   Dense embeddings should distinguish "interval merging" from "sorted list merging"
   even though they share "merge" + "sorted" vocabulary.

2. **Type 2 failures (structural vocabulary overlap):** nested-data tasks retrieving
   flat-data records. Dense embeddings should capture the nesting-structure semantics
   rather than shared "nested", "list", "dict" tokens.

3. **Type 3 failures (repair vocabulary leak):** Only relevant if repair index is used.
   Not tested in v2.17 (champion 99-record index only).

## Expected Family Movements

| Task | Current failure mode | Dense hypothesis |
|---|---|---|
| merge_intervals | lexical collision with merge_sorted | dense separates — predict PASS |
| tree_depth_tuple | structural vocabulary (contested assertion) | uncertain |
| deep_delete | model capability ceiling | dense unlikely to help |
| unflatten_dict | model capability ceiling | dense unlikely to help |
