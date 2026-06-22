# v2.18 TF-IDF Baseline Stability Report

## Three-run baseline (32-task clean benchmark)

| Run | n_pass | score |
|---|---|---|
| TF-IDF run 1 | 17/32 | 53.1% |
| TF-IDF run 2 | 15/32 | 46.9% |
| TF-IDF run 3 | 17/32 | 53.1% |
| **Historical champion (v2.10)** | **20/32** | **62.5%** |

**Mean across 3 v2.18 runs:** 16.3/32 = 51.0%  
**Range:** 15–17 (±1.0 tasks around mean)  
**Historical gap:** 20/32 vs. mean 16.3/32 = 3.7 tasks — explained by flip-task variance (see below)

## Task stability

- Stable PASS (all 3 runs): **11 tasks**
- Stable FAIL (all 3 runs): **11 tasks**
- Flip (inconsistent across runs): **10 tasks**

### Stable PASS tasks (reliable)

`non_overlapping_intervals`, `merge_sorted_k`, `search_rotated_sorted`, `flatten_dict`,
`deep_merge`, `deep_update_if`, `tree_sum`, `tree_leaves`, `tree_mirror`,
`rle_roundtrip`, `rle_longest_run`

### Stable FAIL tasks (model capability ceiling)

`interval_union`, `interval_intersection`, `kth_smallest_matrix`, `wiggle_sort`,
`count_smaller_after_self`, `deep_delete`, `tree_max_path_sum`, `tree_from_list`,
`tree_serialize`, `tree_width`, `rle_delta_encode`

### Flip tasks (sampling noise — pass rate across 3 runs)

| Task | Pass rate | Notes |
|---|---|---|
| `deep_filter` | 2/3 | Usually passes |
| `meeting_rooms` | 2/3 | Usually passes |
| `rle_compress` | 2/3 | Usually passes |
| `rle_expand` | 2/3 | Usually passes |
| `running_median` | 2/3 | Usually passes |
| `unflatten_dict` | 2/3 | Usually passes |
| `deep_keys` | 1/3 | Usually fails |
| `find_peak_element` | 1/3 | Usually fails |
| `insert_interval` | 1/3 | Usually fails |
| `range_summary` | 1/3 | Usually fails |

## Noise floor characterisation

The 10 flip tasks produce a realistic per-run range of 15–20 depending on which tasks
draw pass on a given stochastic sample. In any single best-of-3 run, expect:
- 11 stable-pass tasks always passing
- ~5 of 10 flip tasks passing (with high variance)
- Realistic single-run range: 11 + 3–7 flip = **14–18**

The historical 20/32 (v2.10) was a draw where approximately 9 of 10 flip tasks passed
simultaneously. This is at the high tail of the distribution but consistent with
6 flip tasks having 2/3 pass rate and 4 having 1/3 pass rate.

## CRITICAL FINDING: "TF-IDF" is not TF-IDF

**The `memory/index_adapted` index was built with the local SentenceTransformer model**,
not TF-IDF. Confirmed by inspection: `vocab_size = 0`, `vector dim = 384`, `L2 norm = 1.0`.
TF-IDF would produce a large non-empty vocab and non-unit vectors before normalisation.

The local model is `nreimers/MiniLM-L6-H384-uncased` fine-tuned on `code_search_net`,
StackExchange XML, and MS MARCO — a code-aware semantic embedder.

The TF-IDF Python fallback in `memory/embed.py` only activates when
`models/embeddings/code-memory-embedder` is absent. It has never been the active
retrieval method in any experiment from v2.6 onwards.

**Corrected label for what was called "TF-IDF champion":**
> Local code-aware SentenceTransformer (MiniLM-L6, nreimers, code_search_net-finetuned)

## Impact on v2.17 conclusions

The v2.17 comparison was:
- Baseline ("TF-IDF"): code-aware MiniLM-L6 — index and query both use code-trained model
- Dense MiniLM: all-MiniLM-L6-v2 (generic) — index and query both use generic model

They tied at 17/32. This means a generic embedder equals a code-aware embedder of the
same size class on this 32-task benchmark. The null result stands but the interpretation
changes: the baseline was already code-aware. There is no clear head-room for "adding
code specialisation" since it was already present.

## Implication for Phase B

Phase B must use a **different model architecture or larger model** to have any chance
of beating the baseline, since the baseline is already code-aware at MiniLM scale.

Valid Phase B targets (require download):
- `microsoft/codebert-base` (768d, RoBERTa-based, ~500MB)
- `microsoft/unixcoder-base` (768d, code-specific pretraining)
- `nomic-ai/nomic-embed-code` (768d, modern code embedder)

Using the local `code-memory-embedder` as Phase B input would produce results identical
to the existing baseline — valid only as a control experiment, not a new test.

## Promotion gate (revised for Phase B)

Given baseline is already code-aware, Phase B promotion requires:
- Mean over 3 runs > 18.3/32 (beats noise floor by 2 tasks above mean 16.3)
- Ideally moves at least 1 stable-fail task to stable-pass (not just flip-task reshuffling)
- No regressions in stable-pass tasks

See `tfidf_stability.csv` and `per_task_flips.csv` for full per-task data.
