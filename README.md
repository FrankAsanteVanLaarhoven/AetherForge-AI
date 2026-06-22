<p align="center">
  <img src="docs/splash.jpg" width="100%" alt="AetherForge-AI">
</p>

<h1 align="center">AetherForge-AI</h1>

<p align="center">
  <strong>Memory-augmented local code-agent research</strong><br>
  A forensic investigation into what helps, what fails, and why.
</p>

<p align="center">
  <a href="https://github.com/FrankAsanteVanLaarhoven/AetherForge-AI/releases"><img src="https://img.shields.io/badge/version-2.13-E3B341?style=flat-square" alt="v2.13"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
</p>

---

## What This Project Is

AetherForge-AI is a local memory-augmented code-agent research project built on a
fine-tuned Qwen2.5-Coder-1.5B-Instruct model with an offline verified vector memory.
The system runs entirely on a single consumer GPU (RTX 4080 Super, 16 GB VRAM) with no
external API dependencies.

The project investigates a specific question: **can verified memory retrieval improve a
small locally fine-tuned code-agent, and can targeted repairs or routing strategies
improve it further?**

The answer, after a controlled six-experiment arc: memory retrieval **is** load-bearing
(+17.8 pp), but naïve retraining, global repair-memory promotion, and TF-IDF routing are
not sufficient for robust generalisation. The binding constraint is retrieval relevance.

---

## Final Result Card

| Configuration | Benchmark | Score | Type |
|---|---|---|---|
| Adapter only (no memory) | Frozen 28-task | 18/28 = 64.3% | Clean |
| **Clean champion (memory k=4)** | **Frozen 28-task** | **23/28 = 82.1%** | **Clean** |
| Repair diagnostic index | Frozen 28-task | 27/28 = 96.4% | **Diagnostic only†** |
| Champion on clean generalisation | 32-task clean | 20/32 = 62.5% | Clean |
| Repair on clean generalisation | 32-task clean | 18/32 = 56.2% | Rejected |
| Oracle routing ceiling | 32-task clean | 23/32 = 71.9% | Diagnostic |

> **† Do not misread the 96.4% result.**  
> The 27/28 repair-index result is diagnostic because it targets known frozen-benchmark
> failures. The benchmark is no longer independent for this configuration. It is not
> reported as the clean held-out champion. The clean champion is **23/28 = 82.1%**.

---

## What This Project Proves

- **Verified memory retrieval is load-bearing.** Removing the memory index drops
  performance by 17.8 pp (82.1% → 64.3%). Memory is not decorative — it is required for
  at least 6 tasks that fail entirely without retrieved context.

- **Merge is safe.** `merge_and_unload` applied to the LoRA adapter changes performance
  by ≤3.5 pp (within variance). The merged checkpoint is the reference model.

- **k=4 is the optimal retrieval depth.** k=1 loses −10.7 pp; k=5 loses −3.5 pp.

- **Retraining does not help.** Five retraining configurations were tested. All regressed
  versus the champion, by 17.9–25.0 pp. The original 300-step, 6e-6 LR, agent-only
  training trajectory is not recoverable by continued training.

- **Global repair memory does not generalise.** Adding 4 repair records to the index
  fixes specific known failures but reshuffles TF-IDF retrieval for all tasks, causing
  net regression on 32 clean unseen tasks (56.2% vs 62.5%).

- **TF-IDF routing cannot selectively gate repair.** Family routing, confidence routing,
  and oracle routing were all tested. No routing strategy beats the champion index. The
  root cause is that TF-IDF similarity measures vocabulary overlap, not algorithmic
  relevance — repair records leak across entire algorithm families.

---

## What This Project Does Not Claim

- That 27/28 = 96.4% is a clean held-out champion (it is not)
- That retraining improved the model (it did not)
- That any routing strategy delivers a reliable improvement (none did)
- Generalisation to SWE-bench or production repository tasks (not evaluated)
- Superiority over any other model without co-evaluation on the same tasks
- That the results are production-grade (best-of-3 on 28–32 tasks, single GPU, no API)

---

## Architecture

```
User coding task
        |
        v
Evaluator: system prompt + RETRIEVED_VERIFIED_MEMORY block
        |
        +--- Memory retrieval: memory/index_adapted (99 records)
        |    TF-IDF embedder, k=4, cosine similarity
        |
        v
Merged Qwen2.5-Coder-1.5B champion
(outputs/qwen15b_v27_champion_merged)
        |
        v
execute_code tool call (candidate Python function)
        |
        v
Executor: assertion-based test verification
        |
        v
PASS / FAIL  →  best-of-3  →  result CSV
```

**Index distinction:**

| Index | Path | Records | Status |
|---|---|---|---|
| Champion index | `memory/index_adapted` | 99 | Frozen clean |
| Repair diagnostic | `memory/index_adapted_v29` | 103 | Diagnostic only |

See [`docs/architecture_overview.md`](docs/architecture_overview.md) and
[`docs/architecture_diagram_ascii.md`](docs/architecture_diagram_ascii.md) for full detail.

---

## Benchmark Integrity Finding

During v2.9, we discovered that the `tree_depth_tuple` task in the frozen benchmark
contains a spec-conflicted assertion:

> The prompt expects `tree_depth(((1,2),(3,(4,5)))) == 3`  
> By the stated recursive rule (leaves have depth 1, branches = 1 + max(children)), the correct value is **4**.

A correct implementation fails the assertion. This was documented transparently.
All results report both raw and corrected-audit scores. The corrected champion score is
24/28 = 85.7%. The conservative raw score 23/28 = 82.1% is used throughout.

---

## Dataset and Memory Provenance

| Source | Origin | Type | Status |
|---|---|---|---|
| Frozen 28-task benchmark | Local / constructed | Eval tasks | Clean |
| Champion memory index (99 records) | Local / verified | Memory records | Clean |
| v2.9 repair records (4 records) | Local / targeted | Memory records | Diagnostic |
| 32-task clean benchmark | Local / constructed | Eval tasks | Clean (zero overlap) |
| `data/agent_only_data.jsonl` | Local / curated | Training data | Champion LoRA training |
| Qwen2.5-Coder-1.5B-Instruct weights | HuggingFace Hub | Model weights | Base model |

No HuggingFace datasets were used in the v2.6–v2.13 research arc training or evaluation.
See [`docs/dataset_provenance.md`](docs/dataset_provenance.md) for full detail.

---

## Experiment Timeline (v2.6 – v2.13)

| Version | Experiment | Result | Decision |
|---|---|---|---|
| v2.5 / v2.6 | Data-mixture and trace-ratio retraining | 50–64.3% | Rejected — retraining harmful |
| **v2.7** | **Champion preservation audit** | **82.1% (+17.8 pp memory lift)** | **Champion confirmed** |
| v2.8 | Top-k and prompt-tuning audit | ≤78.6% | Rejected — no improvement |
| v2.9 | Repair memory diagnostic | 96.4% | Diagnostic — benchmark non-independent |
| v2.10 | Clean 32-task generalisation | Champion 62.5%, Repair 56.2% | Repair rejected |
| v2.11 | Routing audit (3 strategies) | All ≤62.5%, oracle 71.9% | All routing rejected |
| v2.12 | Manuscript and reproducibility packet | 7 documents | Evidence packaged |
| v2.13 | Paper draft | Full paper | Evidence closed |

---

## Reproducibility

Verify in 5 minutes:

```bash
make test
make audit-attribution
make summarise-v27-preservation
make summarise-v28
make summarise-v29
make summarise-v210
make summarise-v211
```

Full reproducibility commands: [`results/v212_manuscript_packet/05_reproducibility_commands.md`](results/v212_manuscript_packet/05_reproducibility_commands.md)

The champion model must be present at `outputs/qwen15b_v27_champion_merged`.

---

## Evidence Packet and Manuscript

| Document | Path |
|---|---|
| Executive summary | `results/v212_manuscript_packet/00_executive_summary.md` |
| Experiment timeline | `results/v212_manuscript_packet/01_experiment_timeline.md` |
| Results tables | `results/v212_manuscript_packet/02_main_results_table.md` |
| Failure analysis | `results/v212_manuscript_packet/03_failure_analysis.md` |
| Claim boundary | `results/v212_manuscript_packet/04_claim_boundary.md` |
| Reproducibility commands | `results/v212_manuscript_packet/05_reproducibility_commands.md` |
| Paper draft | `paper/aetherforge_memory_augmented_code_agent_draft.md` |
| Evidence closure | `docs/evidence_closure_certificate.md` |

---

## Forensic Notebook

[`notebooks/AetherForge_Forensic_Memory_Audit.ipynb`](notebooks/AetherForge_Forensic_Memory_Audit.ipynb)

A Kaggle-style forensic audit notebook covering all 6 experiments, result tables,
failure taxonomy charts, decision log, and the certified evidence boundary. No GPU
required — loads existing result files.

---

## Evidence Closure Status

The v2.6–v2.13 research arc is **closed**.

```
Clean champion:     23/28 = 82.1%
Diagnostic repair:  27/28 = 96.4%  (not clean — benchmark non-independent)
Clean generalisation: champion 20/32 = 62.5% beats repair 18/32 = 56.2%
Routing ceiling:    23/32 = 71.9% (oracle, diagnostic only)
Next bottleneck:    operation-aware retrieval relevance
```

The clean champion — merged Qwen2.5-Coder-1.5B + `memory/index_adapted` — is the
stable, reproducible result for this arc.

---

## Future Work

1. **Dense code retrieval** — Replace TF-IDF with CodeBERT or nomic-embed-code to address
   all three identified retrieval failure types (lexical collision, structural overlap,
   repair vocabulary leak).

2. **Operation-aware memory metadata** — Algorithm family tags on memory records to enable
   exact-match routing without TF-IDF false positives.

3. **SWE-bench infrastructure** — Extend the evaluation harness to multi-file repository
   patch generation for real-world relevance.

4. **Larger clean benchmark** — 200+ tasks to reduce sampling variance to below ±1 task.

---

## Repository Structure

```
AetherForge-AI/
├── aetherforge/                         Custom transformer model (pretraining / distillation)
├── scripts/
│   ├── evaluate_code_agent.py           Core evaluation loop (all clean results)
│   ├── finetune_qwen_code_agent.py      LoRA fine-tuning script (champion training)
│   ├── build_memory_index.py            Memory index construction
│   ├── route_v211.py                    Routing audit (v2.11)
│   ├── summarise_v27_preservation.py    Champion audit summariser
│   ├── summarise_v28_champion_system.py Hyperparameter audit summariser
│   ├── summarise_v29_memory_repair.py   Repair memory summariser
│   ├── summarise_v210.py                Clean generalisation summariser
│   └── summarise_v211.py                Routing audit summariser
├── data/
│   ├── heldout_code_agent_tasks.jsonl   Frozen 28-task benchmark
│   └── v210_clean_repair_generalisation_tasks.jsonl  32-task clean benchmark
├── memory/
│   ├── index_adapted/                   Champion memory index (99 records)
│   └── index_adapted_v29/               Repair diagnostic index (103 records)
├── paper/
│   ├── aetherforge_memory_augmented_code_agent_draft.md
│   ├── tables/main_results_table.md
│   └── figures/
├── notebooks/
│   └── AetherForge_Forensic_Memory_Audit.ipynb
├── docs/
│   ├── architecture_overview.md
│   ├── architecture_diagram_ascii.md
│   ├── benchmark_registry.md
│   ├── model_and_memory_registry.md
│   ├── dataset_provenance.md
│   └── evidence_closure_certificate.md
├── results/
│   └── v212_manuscript_packet/          Reproducibility packet (all experiment summaries)
└── Makefile                             All evaluation and summarisation targets
```

---

## Attribution and Authorship

**Author:** Frank Asante Van Laarhoven  
**Contact:** frankleroyvan@gmail.com

This repository contains original research work. All results are reproducible from
the provided source code, data files, and Makefile targets.
