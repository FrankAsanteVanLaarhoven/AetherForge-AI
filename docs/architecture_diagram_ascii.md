# AetherForge Architecture — ASCII Diagrams

## Main Inference Path (Clean Champion)

```
User coding task
        |
        v
+-----------------------------------------------+
|  Evaluator                                    |
|  - Formats system prompt                      |
|  - Injects RETRIEVED_VERIFIED_MEMORY block    |
+-----------------------------------------------+
        |
        | (query text)
        v
+-----------------------------------------------+
|  Memory Retrieval                             |
|  Index: memory/index_adapted (99 records)     |
|  Embedder: code-memory-embedder (ST, 384d, code-aware)  |
|  k=4, cosine similarity                       |
+-----------------------------------------------+
        |
        | (top-4 verified examples)
        v
+-----------------------------------------------+
|  Merged Qwen2.5-Coder-1.5B Champion           |
|  Path: outputs/qwen15b_v27_champion_merged    |
|  LoRA merged via merge_and_unload             |
+-----------------------------------------------+
        |
        | (execute_code tool call)
        v
+-----------------------------------------------+
|  Executor                                     |
|  - Runs candidate Python function             |
|  - Checks test assertions                     |
+-----------------------------------------------+
        |
        v
    PASS / FAIL
        |
        v
+-----------------------------------------------+
|  best-of-3 aggregation                        |
|  PASS if any of 3 generations passes          |
+-----------------------------------------------+
        |
        v
    Result row in eval CSV
```

---

## Repair Path (Diagnostic — Not the Clean Champion)

```
User coding task
        |
        v
+-----------------------------------------------+
|  Memory Retrieval                             |
|  Index: memory/index_adapted_v29 (103 records)|
|  +4 targeted repair records                  |
|  Same embedder and k=4                        |
+-----------------------------------------------+
        |
        | (may retrieve repair record for known failure)
        v
+-----------------------------------------------+
|  Merged Champion Model (same weights)         |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
|  Executor / Verifier                          |
+-----------------------------------------------+
        |
        v
    27/28 = 96.4% on frozen benchmark
    (NOT CLEAN: benchmark not independent —
     repair records target known failures)

    18/32 = 56.2% on 32 clean tasks
    (REJECTED: loses to champion 20/32 = 62.5%)
```

---

## Routing Audit (v2.11 — All Strategies Rejected)

```
Task family classification
        |
        +-- interval_merge? --> repair index (6 tasks)
        |                           family router score: 19/32 = 59.4%
        +-- else            --> champion index (26 tasks)

TF-IDF confidence routing
        |
        +-- repair top-1 >= 0.35 AND margin >= 0.05?
        |       --> repair index (fires on 20/32 tasks — too broad)
        |           conf router score: 20/32 = 62.5% (tied)
        +-- else --> champion index (12 tasks)

Oracle ceiling (diagnostic, not deployable)
        |
        +-- per-task: whichever index passed in v2.10
        |   3 tasks benefit from repair, 9 fail with both
        |   oracle ceiling: 23/32 = 71.9%

Champion index (baseline)
        score: 20/32 = 62.5%
```

---

## Training Path (Historical — Frozen after v2.7)

```
Qwen2.5-Coder-1.5B-Instruct (HF model weights)
        |
        | LoRA fine-tuning
        | data/agent_only_data.jsonl (curated, local)
        | 300 steps, LR 6e-6, rank 16, alpha 32
        v
LoRA adapter
        |
        | merge_and_unload (verified safe, delta <= 3.5 pp)
        v
outputs/qwen15b_v27_champion_merged
        |
        | FROZEN — no further retraining after v2.7
        v
All v2.8–v2.13 experiments use this checkpoint
```

---

## Memory Index Construction Path

```
Verified task-solution pairs (local JSONL)
        |
        | scripts/build_memory_index.py (or equivalent)
        | code-memory-embedder (SentenceTransformer, 384d, code-aware)
        v
memory/index_adapted (99 records) -- CLEAN CHAMPION INDEX
        |
        | + 4 repair records (v2.9)
        | repair records: merge_intervals, median_two_sorted,
        |                 deep_get, tree_depth_tuple
        v
memory/index_adapted_v29 (103 records) -- REPAIR DIAGNOSTIC INDEX
```
