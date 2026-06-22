# v2.18 Retrieval Baseline Stabilisation + Code-Dense Retrieval

**Status:** PHASE A COMPLETE — Phase B pending model download.

## Phase A Results — TF-IDF Baseline Stability (COMPLETE)

| Run | n_pass | score |
|---|---|---|
| Historical champion (v2.10) | 20/32 | 62.5% |
| v2.17 TF-IDF rerun | 17/32 | 53.1% |
| v2.18 run 1 | 17/32 | 53.1% |
| v2.18 run 2 | 15/32 | 46.9% |
| v2.18 run 3 | 17/32 | 53.1% |

**Mean (v2.18 runs 1–3):** 16.3/32 = 51.0%  
**Range:** 15–17  
**Stable PASS tasks:** 11  
**Stable FAIL tasks:** 11  
**Flip tasks:** 10  

**Historical 20/32 explained:** The v2.10 result is within the tail of the flip-task
distribution. With 10 flip tasks (6 with ~67% pass rate, 4 with ~33% pass rate),
a draw of 20/32 is in the upper tail but consistent with sampling variance.

## CRITICAL FINDING: Baseline is already code-aware dense retrieval

The `memory/index_adapted` index (the "TF-IDF champion") was built with a local
SentenceTransformer model, not TF-IDF. Confirmed by index inspection:
`vocab_size = 0`, `dim = 384`, `L2 norm = 1.0`.

The model is `nreimers/MiniLM-L6-H384-uncased` fine-tuned on `code_search_net` and
StackExchange XML. The TF-IDF fallback in `memory/embed.py` only activates when
`models/embeddings/code-memory-embedder` is absent. It has never fired.

**Corrected label for all prior "TF-IDF" experiments:**
> Code-aware dense retrieval — MiniLM-L6 (nreimers/MiniLM-L6-H384-uncased, code_search_net)

**Impact on v2.17 conclusions:** The v2.17 comparison was code-aware MiniLM-L6 (baseline)
vs. generic MiniLM-L6 all-MiniLM-L6-v2 (dense test). They tied. The null result
stands — a generic model does not beat a code-aware model of the same scale.

## Phase B Plan — Code-Dense with Larger Architecture (PENDING)

Phase B remains valid but the target has changed. The baseline is already code-aware
at MiniLM-L6 scale. Phase B must use a **larger or differently-architected** model
to have headroom for improvement.

| Mode | 32-task mean | vs stable baseline | Decision |
|---|---|---|---|
| Code-aware MiniLM-L6 (baseline) | 16.3/32 = 51.0% | 0 | stable baseline |
| Code-dense (new model) | PENDING | PENDING | PENDING |
| Code-hybrid | PENDING | PENDING | PENDING |

**Promotion gate:** Phase B mean over 3 runs must exceed 18.3/32 (baseline mean + noise floor)
AND ideally move at least one stable-fail task to stable-pass.

**Model options for Phase B** (require download, ~500MB each):
- `microsoft/codebert-base` (768d, RoBERTa-based)
- `microsoft/unixcoder-base` (768d)
- `nomic-ai/nomic-embed-code` (768d)

See `tfidf_baseline_report.md` for full per-task stability breakdown.
