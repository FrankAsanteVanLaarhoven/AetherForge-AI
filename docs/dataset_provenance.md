# Dataset and Data Provenance

This document describes the origin, type, and use of every data source in the
AetherForge v2.6–v2.13 research arc.

---

## Data Used in v2.6–v2.13 Experiments

### Training Data

| File | Origin | Type | Used for |
|---|---|---|---|
| `data/agent_only_data.jsonl` | Local — curated manually | Agent-format tool-call examples | Champion LoRA training |
| `data/dev_set_blended.jsonl` | Local — curated | Blended dev set | v2.6 trace-ratio experiments |
| `data/dev_set_data.jsonl` | Local — curated | Dev evaluation | Early eval |
| `data/code_agent_data.jsonl` | Local — generated | Code-agent examples | Earlier training runs |
| `data/execution_traces.jsonl` | Local — generated | Execution trace data | v2.6 trace-blend |

**Hugging Face datasets:** Not used as training data in v2.6–v2.13 experiments. The
`scripts/train_aetherforge.py` pretraining script supports FineWeb streaming, but
the champion LoRA training (`scripts/finetune_qwen_code_agent.py`) uses only local
JSONL data.

### Evaluation Data

| File | Origin | Type | Status |
|---|---|---|---|
| `data/heldout_code_agent_tasks.jsonl` | Local — constructed | 28 hard held-out tasks | Frozen benchmark (clean) |
| `data/v210_clean_repair_generalisation_tasks.jsonl` | Local — constructed | 32 clean generalisation tasks | Clean (zero overlap) |

All evaluation tasks were constructed locally. No tasks were sourced from public
benchmarks such as HumanEval, MBPP, or SWE-bench. No evaluation task was used in
training.

### Memory Records

| Source | Type | Records | Status |
|---|---|---|---|
| `memory/index_adapted` | Locally verified task-solution pairs | 99 | Clean champion index |
| `memory/raw_v29_repair/repair_records.jsonl` | Locally written targeted repairs | 4 | Diagnostic only |
| `memory/index_adapted_v29` | Champion + 4 repair records | 103 | Diagnostic (rejected) |

All memory records are verified: each record's solution was confirmed PASS by execution
before being added to any index.

---

## Hugging Face Resources Used

| Resource | Type | Used for |
|---|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Model weights | Base model for LoRA fine-tuning |
| `models/embeddings/code-memory-embedder` | Local adapted embedder | Memory retrieval |

The Qwen2.5-Coder-1.5B-Instruct model weights are downloaded from HuggingFace Hub
but are not included in this repository. The code-memory-embedder was adapted locally
(not a standard HF dataset).

---

## What Was Not Used in v2.6–v2.13

The following are present in the repository from earlier phases but were not used in
the v2.6–v2.13 research arc:

| Resource | Status |
|---|---|
| `HuggingFaceFW/fineweb` streaming | Only in `train_aetherforge.py` pretraining script, not in champion LoRA |
| `data/synthetic_data.jsonl` | Used in earlier phases; not in v2.6–v2.13 champion training |
| Qwen2.5-VL-7B-Instruct | Used for distillation experiments in earlier phases |
| SWE-bench Lite dataset | Scripts present but no verified results in v2.6–v2.13 arc |

---

## Data Integrity Principles

1. No evaluation task was used in training or in memory record construction.
2. No repair record was added to the frozen 28-task clean benchmark evaluation path.
3. Any benchmark that becomes non-independent (repair records target known failures)
   is immediately reclassified as diagnostic and labelled accordingly.
4. All claims are supported only by results from clean, non-contaminated evaluations.
5. The 96.4% diagnostic result is reported separately and clearly labelled throughout.
