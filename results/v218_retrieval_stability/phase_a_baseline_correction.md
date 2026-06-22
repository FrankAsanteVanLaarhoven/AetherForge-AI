# v2.18 Phase A — Baseline Stability and Retrieval-Framing Correction

## Summary

v2.18 Phase A was designed to stabilise the 32-task retrieval baseline before testing
larger or different dense retrieval models. The main outcome is not a new promoted
retrieval method. The main outcome is a correction to the experimental framing.

The protected memory baseline was previously described as a TF-IDF-style baseline in
some places. Phase A shows that this wording is inaccurate. The protected memory index
uses dense 384-dimensional vectors, not an active TF-IDF fallback path. Therefore,
v2.17 should be interpreted as a same-scale dense-retrieval comparison rather than a
TF-IDF-versus-dense comparison.

No new retrieval method is promoted by Phase A.

## Baseline stability result

Three repeated baseline runs on the clean 32-task benchmark produced:

| Run | Score | Percentage |
|---|---:|---:|
| Run 1 | 17/32 | 53.1% |
| Run 2 | 15/32 | 46.9% |
| Run 3 | 17/32 | 53.1% |
| **Mean** | **16.3/32** | **51.0%** |

The observed range is 15–17 tasks solved. This means the current evaluation has
meaningful sampling variation under the present best-of-3 setting.

## Task stability structure

The 32-task benchmark separates into three groups:

| Group | Count | Interpretation |
|---|---:|---|
| Stable-pass | 11 | Tasks consistently solved by the current system |
| Stable-fail | 11 | Tasks not solved by the current retrieval and agent setup |
| Flip tasks | 10 | Tasks sensitive to sampling and generation variance |

This explains why a single run can move by one or two tasks without proving that
retrieval quality has changed.

**Stable-pass tasks (11):** `non_overlapping_intervals`, `merge_sorted_k`,
`search_rotated_sorted`, `flatten_dict`, `deep_merge`, `deep_update_if`, `tree_sum`,
`tree_leaves`, `tree_mirror`, `rle_roundtrip`, `rle_longest_run`

**Stable-fail tasks (11):** `interval_union`, `interval_intersection`,
`kth_smallest_matrix`, `wiggle_sort`, `count_smaller_after_self`, `deep_delete`,
`tree_max_path_sum`, `tree_from_list`, `tree_serialize`, `tree_width`, `rle_delta_encode`

**Flip tasks (10):** `deep_filter` (2/3), `meeting_rooms` (2/3), `rle_compress` (2/3),
`rle_expand` (2/3), `running_median` (2/3), `unflatten_dict` (2/3), `deep_keys` (1/3),
`find_peak_element` (1/3), `insert_interval` (1/3), `range_summary` (1/3)

## Embedding provenance correction

Phase A inspected the protected memory index (`memory/index_adapted`) and found:

- vector dimension: 384
- vocabulary size: 0
- vectors are L2-normalised
- the TF-IDF fallback path was not active for the protected champion memory

Therefore, the protected baseline should be described as an existing dense code-aware
baseline, not as a true TF-IDF baseline.

**Evidence:** The local SentenceTransformer model (`models/embeddings/code-memory-embedder`)
is `nreimers/MiniLM-L6-H384-uncased` fine-tuned on `code_search_net`, StackExchange XML,
and MS MARCO. The TF-IDF Python fallback in `memory/embed.py` only activates when this
model path is absent. It has never been active in any experiment from v2.6 onwards.

**Correct terminology:**

> protected dense baseline

or

> existing code-aware dense baseline

The phrase "TF-IDF fallback" should be reserved only for the fallback implementation
path in `memory/embed.py`, not for the protected champion memory index.

## Corrected interpretation of v2.17

v2.17 tested a generic MiniLM dense retriever (`all-MiniLM-L6-v2`) and a hybrid
generic MiniLM reranking mode.

The corrected interpretation is:

- generic MiniLM dense retrieval tied the v2.17 baseline run (17/32 each)
- generic MiniLM hybrid retrieval regressed (16/32)
- generic MiniLM did not beat the existing code-aware dense baseline
- this is a null result for generic MiniLM
- this is not evidence that dense retrieval fails globally

The earlier framing of v2.17 as TF-IDF-versus-dense should be corrected. It was a
same-scale code-aware-dense-versus-generic-dense comparison.

## Why the historical 20/32 result is not necessarily degradation

The historical 20/32 result (v2.10) is higher than the three Phase A repeated runs.
However, Phase A identified 10 flip tasks with unstable pass/fail behaviour. A
high-tail run that solves many flip tasks can plausibly reach 20/32 without implying
that the model has degraded later.

The safer interpretation is:

> The 20/32 result remains a valid historical high run, while the repeated-run mean is
> closer to 16.3/32 under the current repeated evaluation protocol.

This does not prove model degradation. The same model and index are used; sampling
variance across best-of-3 draws explains the range.

## Why Phase B must be separate

Phase B would introduce a meaningfully different or larger code-specialised embedding
model. That changes the experimental condition and should not be mixed into the Phase A
baseline-stability result.

Phase B should be evaluated against the stabilised baseline and should report
repeated-run scores, task flips, family-level changes, and any stable-fail tasks
converted into stable-pass tasks.

## Claim boundary

### Supported claims

- The protected memory baseline is code-aware dense retrieval, not an active TF-IDF baseline.
- The current 32-task repeated baseline mean is 16.3/32 = 51.0%.
- The benchmark contains stable-pass (11), stable-fail (11), and flip-task (10) groups.
- Generic MiniLM did not improve the retrieval result in v2.17.
- Hybrid generic MiniLM was not promoted.

### Diagnostic claims

- The historical 20/32 result is plausibly a high-tail run caused by flip-task variance.
- Stable-fail tasks indicate a current capability ceiling under the present retrieval and agent setup.

### Not claimed

- True TF-IDF was beaten by dense retrieval.
- Dense retrieval fails globally.
- The model degraded from v2.10 to v2.18.
- Generic MiniLM should be promoted.
- SWE-bench capability has been measured.
- The system is production-grade.
- The system is generally superior to frontier models.

## Next decision gate

Phase B may test a meaningfully different or larger code-specialised embedding model.
The baseline is already MiniLM-L6 scale code-aware, so Phase B must use a different
architecture (e.g. RoBERTa-based CodeBERT at 768d) or a model with stronger code
pretraining to have headroom for improvement.

Promotion requires improvement beyond the observed flip-task noise band:

- minimum: mean across three runs must exceed 18.3/32 (= 16.3 + 2.0 noise floor)
- strong result: 22/32 or better on a single run
- stronger evidence: at least one stable-fail task becomes stable-pass without causing
  broad family-level regression

Until that happens, the protected dense baseline remains the retained champion.
