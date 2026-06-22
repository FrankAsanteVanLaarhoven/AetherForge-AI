# AetherForge — Reproducibility Commands

All commands assume `~/AetherForge-AI` as the working directory and the
`aetherforge-train` conda environment (torch 2.11, CUDA 12.8).

The champion model must be present at `outputs/qwen15b_v27_champion_merged`.

## 1. Reproduce the clean champion result (23/28 = 82.1%)

```bash
make eval-champion
cat results/eval_champion/summary.md
```

Expected: 23/28 = 82.1% with memory/index_adapted (k=4).

## 2. Reproduce the adapter-only result (18/28 = 64.3%)

```bash
# Run eval without memory
conda run -n aetherforge-train python scripts/evaluate_code_agent.py \
    --hf-model outputs/qwen15b_v27_champion_merged \
    --tasks-file data/heldout_code_agent_tasks.jsonl \
    --mode best_of_n --n 3 \
    --scoring-mode verified_agent \
    --agent-contract strict \
    --stop-after-pass \
    --output outputs/eval_no_memory \
    --verbose
```

Expected: 18/28 = 64.3% (memory lift = +17.8 pp).

## 3. Reproduce v2.9 repair memory inspection

```bash
make inspect-v29-retrieval
cat results/v29_memory_repair/retrieval_inspection.md
```

Shows which records are retrieved (and HELPFUL/HARMFUL) for the 4 repair targets.

## 4. Reproduce v2.9 repair diagnostic eval

```bash
# Build repair memory index
make build-v29-repair-memory
# Run diagnostic on frozen 28-task benchmark
make eval-v29-repair-memory-diagnostic
# Summarise
make summarise-v29
cat results/v29_memory_repair/summary.md
```

Expected: 27/28 = 96.4% (diagnostic; benchmark is no longer independent).

## 5. Reproduce v2.9 clean generalisation eval

```bash
make eval-v29-clean-memory-generalisation
# Results appear in results/v29_memory_repair/
```

Expected: 4/5 = 80.0% on data/v29_clean_generalisation_tasks.jsonl.

## 6. Reproduce v2.10 clean repair-generalisation benchmark

```bash
make eval-v210-clean-champion
make eval-v210-repair-index
make summarise-v210
cat results/v210_clean_repair_generalisation/summary.md
cat results/v210_clean_repair_generalisation/per_family_breakdown.md
```

Expected:
- Champion: 20/32 = 62.5%
- Repair: 18/32 = 56.2%

## 7. Reproduce v2.11 retrieval routing audit

```bash
make route-v211
make eval-v211-family-router
make eval-v211-confidence-router
make eval-v211-oracle-router  # No new eval; uses v2.10 results
make summarise-v211
cat results/v211_retrieval_routing/summary.md
cat results/v211_retrieval_routing/per_task_routing.csv
```

Expected:
- Family router: ~19-21/32 (within variance)
- Confidence router: ~20/32 (tied with champion)
- Oracle ceiling: 23/32 = 71.9%

## 8. View all Makefile targets

```bash
make help
# or
grep "^[a-z].*:" Makefile | grep -v "^#"
```

## 9. Full reproduce from clean state

```bash
# Assumes champion model and raw memory are present
make eval-champion        # verify champion still works
make inspect-v29-retrieval
make build-v29-repair-memory
make build-v29-adapted-repair-index
make eval-v29-repair-memory-diagnostic
make eval-v29-clean-memory-generalisation
make summarise-v29
make eval-v210-clean-champion
make eval-v210-repair-index
make summarise-v210
make route-v211
make eval-v211-family-router
make eval-v211-confidence-router
make summarise-v211
```

## Environment

```bash
conda activate aetherforge-train
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.x True

python -c "import transformers, peft, sentence_transformers; print('deps OK')"
```

## Key paths

| Path | Description |
|---|---|
| `outputs/qwen15b_v27_champion_merged/` | Champion merged model (REQUIRED) |
| `memory/index_adapted/` | Champion memory index (99 records, k=4) |
| `memory/index_adapted_v29/` | Repair index (103 records, locally excluded) |
| `memory/raw_adapted/` | Champion raw records (locally excluded) |
| `memory/raw_v29_repair/repair_records.jsonl` | 4 verified repair records (committed) |
| `data/heldout_code_agent_tasks.jsonl` | Frozen 28-task benchmark |
| `data/v210_clean_repair_generalisation_tasks.jsonl` | 32 clean tasks |
| `data/v29_clean_generalisation_tasks.jsonl` | 5 v2.9 clean tasks |

## Sampling variance note

best-of-3 sampling introduces ±2–3 task variance on the 32-task benchmark.
Individual task outcomes (especially close-to-passing tasks) may differ across runs.
The champion 23/28 = 82.1% is a stable result; individual clean-benchmark scores
(20/32) should be interpreted with ±3 task uncertainty.
