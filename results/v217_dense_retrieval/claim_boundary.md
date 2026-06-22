# v2.17 Claim Boundary

## What is concluded

| Claim | Evidence | Strength |
|---|---|---|
| MiniLM dense retrieval did not beat TF-IDF on clean 32-task benchmark | 17/32 vs 17/32 — tied | Strong (within-run controlled comparison) |
| MiniLM hybrid retrieval slightly underperformed TF-IDF | 16/32 vs 17/32 | Suggestive (within noise floor) |
| MiniLM dense retrieval partially reduces interval-family lexical collisions | interval: 2/6 → 4/6 | Weak (n=6, partially cancelled by sorting regressions) |
| A generic embedder is insufficient for code-task retrieval improvement | Task-level shuffle, zero net delta | Consistent with v2.11 diagnosis |

## What is NOT concluded

| Forbidden claim | Why |
|---|---|
| "Dense retrieval fails" | Only MiniLM tested; code-specialised embedders not evaluated |
| "TF-IDF is better than dense retrieval for code" | Tied result; different embedder may change outcome |
| "Hybrid retrieval hurts" | 1-task regression is within ±2–3 task sampling noise |
| "The champion score dropped" | Baseline variance (17 vs 20) is within stated ±2–3 task range |

## Baseline variance — important caveat

The historical champion score was 20/32 = 62.5% (from v2.10, a separate eval run).
The v2.17 TF-IDF baseline measured 17/32 = 53.1% — a 3-task difference using the
same model, same index, same eval configuration. This is at the upper bound of the
±2–3 task sampling variance stated throughout the paper. It does not indicate model
regression; it is sampling noise at small n (32 tasks, best-of-3).

The fair conclusion is: **all three retrieval modes in v2.17 produce results consistent
with the historical champion within sampling variance.** No mode produced a statistically
distinguishable improvement.

## Promotion decision

Champion retained: TF-IDF with local code-memory-embedder, `memory/index_adapted`, 99 records.
No dense or hybrid configuration is promoted in v2.17.

## Next valid experiment

Test a code-specialised embedding model (CodeBERT, UniXcoder, nomic-embed-code) using
the same evaluation harness. Replace `V217_DENSE_MODEL` in Makefile and rerun:
```
make build-v217-dense-index
make eval-v217-dense-32
make eval-v217-hybrid-32
```
Promotion threshold remains: ≥19/32 (59.4%) on clean 32-task benchmark.
