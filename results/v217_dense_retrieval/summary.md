# v2.17 Dense Retrieval Pilot — Results Summary

**Status:** PENDING — dense index not yet built; model download required.

## Experiment Design

Three retrieval modes evaluated against the champion (TF-IDF) on two benchmarks:
- Frozen 28-task benchmark (same as v2.6–v2.11 experiments)
- Clean 32-task generalisation benchmark (same as v2.10)

| Mode | Description |
|---|---|
| tfidf | Existing champion: word+bigram TF-IDF cosine similarity, k=4 |
| dense | SentenceTransformer all-MiniLM-L6-v2, cosine similarity, k=4 |
| hybrid | TF-IDF top-20 candidates → dense rerank → final k=4 |

**Dense model:** `sentence-transformers/all-MiniLM-L6-v2` (384d, CPU-compatible, ~80MB)  
**Index source:** `memory/index_adapted` (99 verified records, the frozen champion index)  
**Dense index output:** `memory/dense_index_adapted/`

## Promotion Rule

Dense or hybrid must beat the champion on the **clean 32-task benchmark**:
- +2 tasks (62.5% → 68.8%) = strong evidence, promote
- +1 task (62.5% → 65.6%) = weak evidence, not conclusive
- 0 or negative = retrieval mode rejected

## Results

| Mode | 28-task | 32-task | vs champion (32) |
|---|---|---|---|
| tfidf (champion) | 23/28 = 82.1% | 20/32 = 62.5% | — |
| dense | PENDING | PENDING | PENDING |
| hybrid | PENDING | PENDING | PENDING |

## Conclusion

PENDING — run Makefile targets to populate results.

```
make build-v217-dense-index
make eval-v217-tfidf-28
make eval-v217-dense-28
make eval-v217-hybrid-28
make eval-v217-tfidf-32
make eval-v217-dense-32
make eval-v217-hybrid-32
make summarise-v217
```
