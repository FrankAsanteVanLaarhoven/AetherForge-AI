# AetherForge Code Agent — Offline Memory Results

## Summary

AetherForge reached 100% verified-tool pass rate on the 16-task code-agent benchmark using an offline memory-augmented Qwen2.5-Coder-1.5B agent.

The strongest configuration combines:

- Qwen/Qwen2.5-Coder-1.5B-Instruct
- LoRA fine-tuning
- clean agent-only data
- recovery-heavy trajectories
- offline vector memory
- local all-MiniLM-L6-v2 embeddings
- strict verified-agent scoring
- stop-after-pass execution control

## Final Benchmark Result

| Configuration | Pass Rate | Verified Tool Pass | Tool Call Rate | Error Rate | Avg Steps | Repeated Calls | Unnecessary Retry |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base only | 75.0% | 75.0% | 93.8% | 12.5% | 4.69 | 0 | 0 |
| Memory only | 100.0% | 100.0% | 100.0% | 0.0% | 4.38 | 0 | 0 |
| Clean memory 25-step pilot | 100.0% | 100.0% | 100.0% | 0.0% | 2.06 | 0 | 0 |
| Memory 300-step single | 100.0% | 100.0% | 100.0% | 0.0% | 2.06 | 0 | 0 |
| Memory 300-step best-of-3 | 100.0% | 100.0% | 100.0% | 0.0% | 2.06 | 0 | 0 |

## Interpretation

This result validates offline vector memory plus recovery-focused agent training on the known verified benchmark.

It should not yet be reported as broad unseen-task generalisation because the memory bank includes extracted verified examples from previous evaluation logs.

Next step: held-out and recovery-stress evaluation.
