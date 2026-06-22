# AetherForge System Architecture

## Overview

AetherForge is a local memory-augmented code-agent. The system runs entirely on a single
consumer GPU (RTX 4080 Super, 16 GB VRAM) with no external API dependencies. The core
loop is: receive a coding task, retrieve verified similar examples from a local memory
index, generate candidate code via tool use, execute and verify the candidate, and record
the result.

## Components

### 1. Base Model — Merged Champion

**Path:** `outputs/qwen15b_v27_champion_merged`

Qwen2.5-Coder-1.5B-Instruct fine-tuned with LoRA (rank 16, alpha 32, 300 steps,
learning rate 6e-6) on a curated agent-only dataset. The adapter is merged at evaluation
time using `merge_and_unload`. The merge is verified safe: unmerged and merged adapters
differ by ≤3.5 pp on the frozen benchmark (within variance).

The training dataset is local and agent-specific: `data/agent_only_data.jsonl`. It
contains `execute_code` tool-call examples where the model produces Python code, an
executor runs it, and the observation confirms correctness. This teaches the tool-call
format without injecting execution traces that shift output distribution.

### 2. Memory System

**Champion index:** `memory/index_adapted` (99 records)
**Repair index:** `memory/index_adapted_v29` (103 records = champion + 4 repair records)

Each memory record contains:
- `query_text`: Task description used for similarity matching
- `corrected_tool_call`: A verified working `execute_code` call
- Status: All records confirmed PASS by execution

**Embedder:** Local TF-IDF-like sentence embedder at `models/embeddings/code-memory-embedder`
**Retrieval:** Cosine similarity, top-k=4 (ablation-selected), minimum score threshold 0.0
**Injection:** Retrieved records are formatted as `RETRIEVED_VERIFIED_MEMORY` in the system prompt

### 3. Evaluation Protocol

**Script:** `scripts/evaluate_code_agent.py`
**Mode:** `best_of_n` (n=3)
**Scoring:** `verified_agent` — the model must produce an `execute_code` tool call whose
execution returns `PASS` via test assertion
**Agent contract:** `strict` — no direct answer extraction fallback
**Stop-after-pass:** Enabled — the first passing generation is accepted

Sampling variance: ±2–3 tasks on a 32-task benchmark at best-of-3.

### 4. Retrieval Path (Inference)

```
User coding task (natural language)
        ↓
Evaluator formats: system prompt + RETRIEVED_VERIFIED_MEMORY block + task prompt
        ↓
Merged Qwen2.5-Coder-1.5B champion
        ↓
Output: execute_code tool call containing candidate Python function
        ↓
Executor: runs candidate against test assertions
        ↓
PASS / FAIL
        ↓
best-of-3 result → CSV row in eval output
```

### 5. Memory Retrieval Detail

```
Task prompt text
        ↓
code-memory-embedder → dense vector
        ↓
Cosine similarity against all 99 records in memory/index_adapted
        ↓
Top-4 records above min_score threshold
        ↓
Injected into system prompt as verified examples
```

### 6. Index Distinction

| Index | Path | Records | Status |
|---|---|---|---|
| Champion index | `memory/index_adapted` | 99 | Clean — used for all clean results |
| Repair index | `memory/index_adapted_v29` | 103 | Diagnostic — adds 4 targeted repair records |

The repair index was created by adding 4 records targeting known frozen-benchmark
failures. It reaches 27/28 = 96.4% diagnostically but fails to generalise globally
(18/32 = 56.2% on clean 32-task benchmark vs 20/32 for champion index).

### 7. Routing Experiments (v2.11, concluded)

Three routing strategies were tested to selectively apply the repair index:

| Strategy | Logic | Result |
|---|---|---|
| Family router | interval_merge family → repair, others → champion | 19/32 = 59.4% (−3.1 pp) |
| Confidence router | TF-IDF margin ≥ 0.05 → repair, else → champion | 20/32 = 62.5% (0 pp) |
| Oracle ceiling | Perfect per-task routing | 23/32 = 71.9% (+9.4 pp) |

All three routing strategies fail to improve over champion. TF-IDF similarity measures
vocabulary overlap, not algorithm-level repair relevance. The oracle ceiling shows that
only 3 of 32 tasks benefit from repair routing; 9 tasks fail regardless of routing.

## What the Architecture Does Not Include

- No SWE-bench multi-file patch generation in the completed v2.6–v2.13 arc
- No dense semantic retrieval (CodeBERT/UniXcoder) — tested only with TF-IDF
- No production serving of the champion model (evaluation-only)
- No cloud inference — fully local

## Hardware

| Component | Specification |
|---|---|
| GPU | RTX 4080 Super (16 GB VRAM) |
| Model VRAM (merged, fp16) | ~3.2 GB |
| Inference batch size | 1 |
| Total VRAM during eval | ~5–7 GB |
