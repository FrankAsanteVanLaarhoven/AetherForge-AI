# Offline Vector Memory Results — 2026-06-20

## Setup

- Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- LoRA: outputs/qwen15b_balanced_2k_fixloop_25/final
- Memory: offline vector memory
- Records: 70 verified records
  - 8 hand-curated seed memories
  - 62 extracted verified memories
- Embedding backend: TF-IDF fallback
- Scoring: verified_agent
- Fallback scoring: disabled / 0.0%
- Runtime proof required: execute_code must produce OBSERVATION: PASS

## Results

| Mode | Pass Rate | Verified Tool Pass | Notes |
|---|---:|---:|---|
| Base only | 75.0% | 75.0% | No memory |
| Memory only | 100.0% | 100.0% | 70-record memory |
| LoRA + memory single | 93.8% | 93.8% | One failed task: palindrome |
| LoRA + memory best-of-3 | 100.0% | 100.0% | All 16 tasks passed |

## Safety Metrics

- Fallback pass: 0.0%
- Invalid JSON calls: 0
- No assert calls: 0
- No-output observations: 0
- Unknown tool calls: 0
- Recovered after ERROR: 1 in best-of-3

## Interpretation

The result validates offline vector memory as a useful verified recall layer for an air-gapped code agent. The 70-record memory bank significantly improves performance on the known benchmark. Because extracted memories were built from previous evaluations, this should be reported as an adaptive memory / continual improvement result, not as an unseen-task generalisation result.

## Next Step

Run a held-out benchmark with tasks that are not present in memory or previous eval logs.
