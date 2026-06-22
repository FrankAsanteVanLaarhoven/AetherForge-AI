# AetherForge — Executive Summary

## System

AetherForge is a memory-augmented code-agent system: a fine-tuned Qwen2.5-Coder-1.5B-Instruct
adapter (LoRA, 300 steps, 6e-6 LR, agent-only curated data) combined with an offline verified
vector memory (TF-IDF-like embeddings, cosine similarity, JSONL records, k=4 retrieval).

The system operates without cloud APIs. All inference, retrieval, and evaluation run locally
on consumer hardware (RTX 4080 Super, 16 GB VRAM).

## Final champion

| Component | Description |
|---|---|
| Model | `outputs/qwen15b_v27_champion_merged` (merged LoRA) |
| Memory index | `memory/index_adapted` (99 verified records, k=4) |
| Frozen benchmark | 23/28 = 82.1% (28-task hard held-out set) |
| Memory lift | +17.8 pp over adapter-only (64.3% → 82.1%) |

## Evidence trail (v2.6 – v2.11)

| Version | Type | Key result |
|---|---|---|
| v2.6 | Negative (retraining) | Trace-gating at 0-25% does not recover champion; best 57.1% |
| v2.7 | Positive (system audit) | Merge is safe; memory adds +17.8 pp; adapter-only = 64.3% |
| v2.8 | Negative (hyperparameter) | Top-k tuning and direct-answer prompting do not beat champion |
| v2.9 | Diagnostic (repair memory) | Repair memory fixes 4 known failures → 27/28 = 96.4% (diagnostic only) |
| v2.10 | Negative (global repair) | Repair index loses on 32 clean tasks: 18/32 vs 20/32 champion |
| v2.11 | Negative (routing) | TF-IDF routing cannot selectively recover repair gains; oracle ceiling 23/32 |

## Main claim

Verified memory retrieval is load-bearing for this code-agent: it contributes +17.8 pp
and enables 6 task categories that fail without it. Retraining, global repair-memory
promotion, and TF-IDF routing do not improve over the original champion.
The retrieval relevance signal — not the memory content — is the binding constraint.

## Allowed claims

- Adapter-only: 64.3% on 28-task frozen benchmark.
- Adapter + memory: 23/28 = 82.1%.
- Memory lift: +17.8 pp, concentrated in hard tasks.
- Repair memory fixes known failures diagnostically (27/28) but does not generalise cleanly.

## Not allowed claims

- SWE-bench or production-grade reliability.
- The 96.4% diagnostic as a clean held-out champion.
- General superiority over other models or systems.
- Any claim without a corresponding held-out evaluation.
