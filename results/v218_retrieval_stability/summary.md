# v2.18 Retrieval Baseline Stabilisation + Code-Dense Retrieval

**Status:** IN PROGRESS

## Motivation

v2.17 measured TF-IDF at 17/32 = 53.1% vs. historical champion 20/32 = 62.5%.
The 3-task gap is at the upper bound of the stated ±2–3 task sampling variance.
Before testing code-specialised embedders, the baseline must be characterised with
repeated runs so that any improvement can be evaluated against a known noise floor.

## Phase A — TF-IDF Baseline Stability

Three repeated TF-IDF runs on the clean 32-task benchmark.
Same model, same index, same configuration — only random sampling varies.

| Run | n_pass | score | Status |
|---|---|---|---|
| Historical champion (v2.10) | 20/32 | 62.5% | reference |
| v2.17 TF-IDF rerun | 17/32 | 53.1% | −3 from historical |
| v2.18 run 1 | PENDING | PENDING | — |
| v2.18 run 2 | PENDING | PENDING | — |
| v2.18 run 3 | PENDING | PENDING | — |

**Mean / range:** PENDING  
**Stable-pass tasks:** PENDING  
**Stable-fail tasks:** PENDING  
**Flip tasks (sampling noise):** PENDING

See `tfidf_baseline_report.md` after running `make summarise-v218-tfidf-stability`.

## Phase B — Code-Specialised Dense Retrieval

Only after Phase A determines the noise floor.

| Mode | 32-task | vs stable TF-IDF | Decision |
|---|---|---|---|
| TF-IDF (stabilised mean) | PENDING | 0 | baseline |
| Code-dense | PENDING | PENDING | PENDING |
| Code-hybrid | PENDING | PENDING | PENDING |

**Promotion rule:** code-dense or hybrid must beat the stabilised TF-IDF mean by
more than the measured noise floor, and reach ≥22/32 for strong evidence.

## Claim Boundaries

- Do not promote any configuration unless it beats stabilised TF-IDF on the 32-task benchmark.
- Do not claim generalisation beyond these 32 tasks.
- Do not claim a code embedder outperforms TF-IDF unless the 32-task evidence supports it.
- If code-dense improves one family but regresses others, record as family-specific only.
- `memory/index_adapted` remains champion until cleanly beaten.
