# Model and Memory Registry

This document registers all model checkpoints and memory indexes used in the AetherForge
v2.6–v2.13 research arc.

---

## Models

### Champion Model

| Property | Value |
|---|---|
| Path | `outputs/qwen15b_v27_champion_merged` |
| Base | Qwen2.5-Coder-1.5B-Instruct (HuggingFace Hub) |
| Adaptation | LoRA rank 16, alpha 32, 300 steps, LR 6e-6 |
| Training data | `data/agent_only_data.jsonl` (local, curated) |
| Merge status | `merge_and_unload` applied — standalone checkpoint |
| VRAM (fp16) | ~3.2 GB |
| Status | **FROZEN** — no further training after v2.7 audit |

**Merge safety:** Verified. Unmerged and merged adapters differ by ≤3.5 pp on the
frozen 28-task benchmark, within sampling variance.

### Rejected Retrained Models (v2.5, v2.6)

| Experiment | LR | Steps | Score | Decision |
|---|---|---|---|---|
| Merge + fresh LoRA | 5e-6 | 350 | 64.3% | Rejected |
| v2.5 clean foundation | 2e-5 | 300 | 53.6% | Rejected |
| v2.6 traces=0% | 2e-5 | 300 | 57.1% | Rejected |
| v2.6 traces=10% | 2e-5 | 300 | 50.0% | Rejected |
| v2.6 traces=25% | 2e-5 | 300 | 53.6% | Rejected |

All retrained models are not committed to the repository (large files excluded by
`.gitignore`). Only the champion merged checkpoint is the reference model.

---

## Memory Indexes

### Champion Index

| Property | Value |
|---|---|
| Path | `memory/index_adapted` |
| Records | 99 |
| Embedder | `models/embeddings/code-memory-embedder` (local code-aware SentenceTransformer, MiniLM-L6, code_search_net) |
| Status | **FROZEN CLEAN CHAMPION INDEX** |
| Used in | All clean results (v2.7, v2.8, v2.10, v2.11) |

All 99 records are verified task-solution pairs. Each record has been confirmed PASS
by execution before indexing.

### Repair Diagnostic Index

| Property | Value |
|---|---|
| Path | `memory/index_adapted_v29` |
| Records | 103 (= champion + 4 repair records) |
| Repair records | `memory/raw_v29_repair/repair_records.jsonl` |
| Status | Diagnostic — not promoted |
| Used in | v2.9 diagnostic, v2.10 clean generalisation, v2.11 routing audit |

The 4 repair records target: `merge_intervals`, `median_two_sorted`, `deep_get`,
`tree_depth_tuple`. Adding them to the global index reshuffles TF-IDF retrieval for
all tasks, not just the 4 targets. This causes the repair index to lose to champion on
32 clean generalisation tasks (18/32 vs 20/32).

### Promotion Decision

The repair index is **not promoted**. The champion index remains the recommended
configuration. Reason: repair vocabulary leaks to entire algorithm families via TF-IDF
similarity, causing regressions on tasks that pass with the champion index.

---

## Embedder

| Property | Value |
|---|---|
| Path | `models/embeddings/code-memory-embedder` |
| Type | Local code-aware SentenceTransformer (MiniLM-L6, nreimers/MiniLM-L6-H384-uncased) |
| Similarity | Cosine |
| Retrieval k | 4 (ablation-selected in v2.8) |
| Min score | 0.0 (no threshold; top-k always returned) |

The embedder is a locally trained/adapted model. It was not replaced in any v2.6–v2.13
experiment. Dense retrieval (CodeBERT/UniXcoder) is identified as future work and was
not evaluated in this arc.
