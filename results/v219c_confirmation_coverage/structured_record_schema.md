# v2.19 Structured Memory Record Schema

Deterministically derived from the protected champion index (`memory/index_adapted`, 99 verified records). No model weights, no network.
Original injection fields (`task`, `corrected_tool_call`, `observation`, ...) are
preserved verbatim so the prompt block matches the baseline; only embedding,
retrieval, and reranking change.

## Fields

| Field | Meaning |
|---|---|
| record_id | Source record id |
| task_family | Algorithm family (deterministic classifier) |
| task_signature | Primary `name(args)` from the verified solution |
| failure_mode | Recovered failure type (from the source record) |
| retrieval_cues | Deterministic keyword cues (task + identifiers) |
| verified_solution | Parsed execute_code payload |
| minimal_test | Assert/print lines extracted from the solution |
| tool_trace_summary | Compact tool-trace summary |
| why_this_memory_helps | Templated relevance rationale |
| source_record | Provenance pointer |

## Family distribution (121 records)

| Family | Count |
|---|---:|
| math | 30 |
| sort | 17 |
| string | 16 |
| dict | 12 |
| misc | 11 |
| search | 8 |
| tree | 8 |
| interval | 7 |
| rle | 5 |
| cache | 3 |
| graph | 3 |
| matrix | 1 |

## Example (one record, fields abbreviated)

```json
{
  "record_id": "d2121d76-063b-4b6e-a181-d34ec0774bf0",
  "task_family": "math",
  "task_signature": "factorial()",
  "failure_mode": "none",
  "retrieval_cues": [
    "factorial",
    "iterative",
    "recursion"
  ],
  "verified_solution": "from math import factorial\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800\nprint('PASS')",
  "minimal_test": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800\nprint('PASS')",
  "tool_trace_summary": "execute_code -> OBSERVATION: PASS",
  "why_this_memory_helps": "Verified math example defining `factorial()`; relevant when the task needs factorial, iterative, recursion.",
  "source_record": "outputs/code_agent_eval_fixloop_50_verified_agent/single.csv"
}
```
