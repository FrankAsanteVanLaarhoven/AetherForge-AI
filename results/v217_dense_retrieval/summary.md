# v2.17 Dense Retrieval Pilot — Results Summary

**Status:** COMPLETE — 32-task clean benchmark evaluated; 28-task pending.  
**Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384d, generic, CPU/GPU)  
**Index source:** `memory/index_adapted` (99 verified records, frozen champion index)

## Results — Clean 32-Task Generalisation Benchmark

| Mode | 32-task n_pass | 32-task score | vs TF-IDF (this run) | vs historical champion |
|---|---|---|---|---|
| TF-IDF (historical champion, v2.10) | 20/32 | 62.5% | +3 | 0 |
| **TF-IDF (v2.17 baseline)** | **17/32** | **53.1%** | 0 | −3 |
| Dense — MiniLM | 17/32 | 53.1% | 0 | −3 |
| Hybrid — MiniLM | 16/32 | 50.0% | −1 | −4 |

**Baseline variance note:** The TF-IDF baseline in this run is 17/32 = 53.1%, versus
the historical champion score of 20/32 = 62.5% from v2.10. The 3-task difference is
at the upper bound of the stated ±2–3 task sampling variance at best-of-3. The
comparison between retrieval modes is fair within this run; all three modes used the
same model and sampling configuration.

## Promotion Decision

**Neither dense nor hybrid is promoted.** Promotion requires beating the v2.17 TF-IDF
baseline by +2 tasks (≥19/32 = 59.4%). Both modes tied or regressed.

| Threshold | Required | Dense | Hybrid | Met? |
|---|---|---|---|---|
| Strong evidence | ≥19/32 = 59.4% | 17/32 | 16/32 | No / No |
| Tie | 17/32 = 53.1% | 17/32 | 16/32 | Tie / No |
| Regression | <17/32 | — | 16/32 | — / Yes |

**Champion retained:** TF-IDF (local code-memory-embedder), `memory/index_adapted`, 99 records.

## Conclusion

MiniLM dense retrieval did not beat TF-IDF on this benchmark. This is a **null result**
for MiniLM specifically, not a conclusion that dense retrieval fails in general.

`all-MiniLM-L6-v2` is a generic semantic embedder — it was not trained on code. It encodes
surface meaning well but cannot distinguish between algorithmically-different code patterns
that share natural-language phrasing (e.g. "find intervals that", "sort elements by").

The per-task comparison reveals a task-shuffle rather than a systematic improvement:
dense gained 4 tasks (interval family: meeting_rooms, range_summary; heap: running_median;
dict: unflatten_dict) but lost 4 others (sorting: merge_sorted_k, find_peak_element;
dict: deep_keys; rle: rle_compress). Net delta: zero.

## What MiniLM partially showed (interval family)

The interval family moved from 2/6 (TF-IDF) to 4/6 (dense), suggesting dense embeddings
do help with the Type 1 lexical collision problem for this family. However this gain was
cancelled by sorting/heap regressions, confirming that a generic embedder introduces
its own false-positive pattern on code tasks.

## Three retrieval failure types — status after v2.17

| Failure type | Root cause | MiniLM result |
|---|---|---|
| Type 1 — surface lexical collision | "merge" / "sorted" tokens | Partial: interval +2, sorting −2 |
| Type 2 — structural vocabulary overlap | "nested"/"list"/"depth" tokens | Inconclusive: task-level noise |
| Type 3 — repair vocabulary leak | Not tested in v2.17 (champion index only) | N/A |

## Next step

Test a **code-specialised embedder**: CodeBERT, UniXcoder, or `nomic-embed-code`.
These are trained to distinguish algorithmic patterns from vocabulary overlap. The oracle
ceiling (23/32 = 71.9%) shows 3 tasks are theoretically recoverable with perfect routing.
A code embedder is the most direct path to reaching that ceiling.

```bash
# To test a code embedder (when available locally):
make build-v217-dense-index  # update V217_DENSE_MODEL in Makefile
make eval-v217-dense-32
make eval-v217-hybrid-32
make summarise-v217
```

## 28-Task Frozen Benchmark

Not yet evaluated — not required for the promotion decision (32-task is the clean test).
Run `make eval-v217-tfidf-28 eval-v217-dense-28 eval-v217-hybrid-28` if needed for
internal diagnostic comparison.
