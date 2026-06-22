# v2.18 Phase B — Per-Family Breakdown

Average tasks solved per family (n_pass summed over 3 runs ÷ 3). Families are
derived from the task-id prefix token. Baseline = protected code-aware MiniLM
dense (384d). Hybrid stage-1 shortlist is also code-aware MiniLM dense, not TF-IDF.

| Family | Tasks | Baseline | Code-dense | Δ dense | Code-hybrid | Δ hybrid |
|---|---:|---:|---:|---:|---:|---:|
| count | 1 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| deep | 5 | 3.00 | 2.67 | -0.33 | 3.33 | +0.33 |
| find | 1 | 0.33 | 1.00 | +0.67 | 0.33 | +0.00 |
| flatten | 1 | 1.00 | 1.00 | +0.00 | 1.00 | +0.00 |
| insert | 1 | 0.33 | 0.00 | -0.33 | 0.00 | -0.33 |
| interval | 2 | 0.00 | 0.33 | +0.33 | 0.00 | +0.00 |
| kth | 1 | 0.00 | 0.67 | +0.67 | 0.33 | +0.33 |
| meeting | 1 | 0.67 | 0.33 | -0.33 | 0.33 | -0.33 |
| merge | 1 | 1.00 | 1.00 | +0.00 | 1.00 | +0.00 |
| non | 1 | 1.00 | 1.00 | +0.00 | 1.00 | +0.00 |
| range | 1 | 0.33 | 1.00 | +0.67 | 1.00 | +0.67 |
| rle | 5 | 3.33 | 4.33 | +1.00 | 4.33 | +1.00 |
| running | 1 | 0.67 | 1.00 | +0.33 | 0.67 | +0.00 |
| search | 1 | 1.00 | 1.00 | +0.00 | 1.00 | +0.00 |
| tree | 7 | 3.00 | 2.00 | -1.00 | 3.00 | +0.00 |
| unflatten | 1 | 0.67 | 1.00 | +0.33 | 0.33 | -0.33 |
| wiggle | 1 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
