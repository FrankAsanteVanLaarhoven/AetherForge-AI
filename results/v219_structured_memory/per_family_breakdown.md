# v2.19 Per-Family Breakdown

Average tasks solved per family (n_pass summed over runs ÷ run count). Baseline =
protected code-aware MiniLM dense (384d). Structured modes hold the SAME encoder and
vary only record structure + multi-view query + reranking.

| Family | Tasks | Baseline | structured_dense | structured_hybrid |
|---|---:|---:|---:|---:|
| dict | 7 | 4.67 | 4.67 (+0.00) | 4.67 (+0.00) |
| interval | 6 | 2.33 | 3.67 (+1.33) | 4.00 (+1.67) |
| matrix | 1 | 0.00 | 1.00 (+1.00) | 0.00 (+0.00) |
| rle | 5 | 3.33 | 3.00 (-0.33) | 3.67 (+0.33) |
| search | 5 | 2.00 | 1.67 (-0.33) | 2.67 (+0.67) |
| sort | 1 | 1.00 | 1.00 (+0.00) | 1.00 (+0.00) |
| tree | 7 | 3.00 | 3.00 (+0.00) | 3.33 (+0.33) |
