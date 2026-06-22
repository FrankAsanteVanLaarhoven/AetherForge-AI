# v2.17 Claim Boundary

## What this experiment can prove

| Claim | Evidence required | Strength |
|---|---|---|
| Dense retrieval beats TF-IDF on clean tasks | +2 tasks on 32-task benchmark (68.8%+) | Strong |
| Dense retrieval matches TF-IDF | ±1 task, same or better on clean 32 | Neutral — no regression |
| Dense retrieval reduces lexical false positives | Per-family breakdown shows Type 1/2 failures resolved | Mechanistic |
| Hybrid retrieval adds over pure dense | Hybrid > dense on 32-task by 1+ task | Weak (noise floor ≥2) |

## What this experiment cannot prove

- Dense retrieval generalises to tasks outside these 32 families
- Dense retrieval helps at model capability ceiling (9/32 tasks failing with both indexes)
- The oracle ceiling (71.9%) is reachable with dense retrieval alone
- Results on SWE-bench or multi-file patch tasks

## Promotion decision rule

- **Promote dense as new champion:** dense 32-task ≥ 22/32 (68.8%), no regression on 28-task
- **Promote hybrid as new champion:** hybrid 32-task ≥ 22/32 (68.8%), no regression on 28-task
- **Report as null result:** best dense/hybrid on 32-task ≤ 21/32 (65.6%)
- **Report as regression:** dense/hybrid 32-task < 20/32 (62.5%) — TF-IDF champion retained

## Diagnostic classification rule

If dense or hybrid index was built using any records from `index_adapted_v29` (the repair index),
all results from that configuration are classified as **diagnostic** and must not be reported
as a clean champion. Only results built exclusively from `index_adapted` (99 records) are clean.
