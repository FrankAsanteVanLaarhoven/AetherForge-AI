# v2.19b Per-Family Breakdown

Average tasks solved per family (n_pass ÷ runs). Baseline = protected code-aware
MiniLM dense. v219b = same encoder + structured retrieval over the COMBINED pool
(99 + 16 family-targeted records). Families interval/tree/rle/dict received new
verified same-family-different-task coverage.

| Family | Tasks | Baseline | v219b structured-dense | Δ |
|---|---:|---:|---:|---:|
| dict | 7 | 4.67 | 5.33 | +0.67 |
| interval | 6 | 2.33 | 4.00 | +1.67 |
| matrix | 1 | 0.00 | 0.00 | +0.00 |
| rle | 5 | 3.33 | 3.67 | +0.33 |
| search | 5 | 2.00 | 2.67 | +0.67 |
| sort | 1 | 1.00 | 1.00 | +0.00 |
| tree | 7 | 3.00 | 2.67 | -0.33 |
