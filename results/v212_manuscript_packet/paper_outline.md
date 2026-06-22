# AetherForge — Paper Outline

**Working title:**
*Memory-Augmented Code-Agent Adaptation: When Retrieval Helps, When Retraining Hurts,
and Why Routing Is the Bottleneck*

## Abstract (draft)

We present a controlled empirical study of AetherForge, a memory-augmented code-agent
built on a fine-tuned Qwen2.5-Coder-1.5B-Instruct adapter with an offline verified
vector memory. Our main finding is that verified retrieval is load-bearing for this
system: removing memory drops performance by 17.8 pp on our frozen 28-task benchmark
(82.1% → 64.3%). However, retraining, global repair-memory promotion, and TF-IDF
confidence routing all fail to improve over the base configuration. We identify retrieval
noise — high-frequency vocabulary overlap causing wrong-pattern context injection — as the
primary barrier to further improvement. The oracle upper bound for any index-selection
routing scheme is 23/32 = 71.9% on a 32-task generalisation benchmark, leaving 9 tasks
unsolvable regardless of routing. Our evidence trail includes five controlled experiments
with positive results, negative results, and failure analyses, providing a reproducible
reference point for offline memory-augmented code agents.

## 1. Introduction

- Local LLM deployment for code generation: motivation and constraints
- The memory-augmented agent design: offline retrieval, verified examples, no API calls
- Research question: when does retrieved verified memory help, and when is it insufficient?
- Contribution summary: five controlled experiments, one positive finding (memory lift),
  four negative findings (retraining, global repair, TF-IDF routing), and identified
  root cause (retrieval relevance)

## 2. System Design

### 2.1 Model
- Qwen2.5-Coder-1.5B-Instruct base
- LoRA adapter: 300 steps, 6e-6 LR, agent-only curated data
- merge_and_unload at inference time

### 2.2 Memory
- Offline TF-IDF-like embeddings (local sentence embedder)
- JSONL records: query_text, corrected_tool_call, observation (PASS), verified=True
- Cosine similarity retrieval, k=4 optimal
- Memory format: RETRIEVED_VERIFIED_MEMORY block injected into system prompt

### 2.3 Evaluation protocol
- 28-task frozen held-out benchmark (hard single-function Python tasks)
- best-of-3 sampling, verified_agent scoring, strict agent contract
- Sampling variance characterised at ±2–3 tasks on 32-task scale

## 3. Main Findings

### 3.1 Memory is load-bearing (v2.7)

| Config | Score |
|---|---|
| Adapter + memory (k=4) | 23/28 = 82.1% |
| Adapter only | 18/28 = 64.3% |

Memory lift = +17.8 pp. The lift is concentrated in hard tasks:
6 tasks pass only with memory; 2 are hurt (retrieval noise).
merge_and_unload is safe: merged vs unmerged within noise.

### 3.2 Retraining does not improve the champion (v2.5 / v2.6)

Five experiments across 0–25% trace ratios, all below 60%.
The original adapter's performance depends on its training trajectory.
Root cause: higher LR + more data overwrites the generalisation properties
of the original agent-only fine-tuning.

### 3.3 Targeted repair memory fixes known failures diagnostically (v2.9)

4 verified repair examples fix all 4 target tasks → 27/28 = 96.4%.
This is diagnostic: the benchmark is no longer independent after targeted repair.
Clean transfer test: 4/5 = 80.0% (5 untouched tasks, early positive signal).

### 3.4 Global repair memory does not generalise (v2.10)

32 clean untouched tasks, 5 algorithm families.
Champion 20/32 = 62.5%, repair index 18/32 = 56.2%.
Root cause: repair records contaminate retrieval for entire families
through vocabulary overlap, not just the specific repair targets.

### 3.5 TF-IDF routing cannot selectively apply repair memory (v2.11)

Family router: 19/32 = 59.4% (within variance).
Confidence router: 20/32 = 62.5% (tied with champion; gains cancel losses).
Oracle ceiling: 23/32 = 71.9% (+9.4 pp, 3 tasks recoverable).

Root cause: TF-IDF similarity margin measures vocabulary family overlap,
not algorithm-specific repair relevance. Repair records score 0.59–0.82
against entire families, not just repair targets.

## 4. Analysis

### 4.1 Why memory is load-bearing but not scalable with the current retrieval signal

The verify-pass protocol ensures all memory examples are correct.
The TF-IDF similarity cannot distinguish algorithm similarity from vocabulary similarity.
This creates a ceiling: the retrieval system can find relevant examples for common
patterns (count_islands, gcd_lcm, topo_sort) but fails for cases where the query
and a wrong-pattern record share high-frequency tokens.

### 4.2 Retrieval noise taxonomy

Three failure types identified (see also 03_failure_analysis.md):
1. Surface lexical collision (merge_intervals vs merge_sorted records)
2. Structural vocabulary overlap (flatten records vs tree_depth)
3. Repair vocabulary leak (repair records dominate entire algorithm families)

### 4.3 The oracle gap and its implications

Oracle ceiling 23/32 = 71.9% leaves 9 tasks unreachable by any routing.
Those 9 tasks require either:
(a) model-level improvement (more capable base model, or task-specific fine-tuning)
(b) additional verified memory patterns covering those algorithms
(c) multi-step reasoning loops not supported in the current eval framework

The 3-task routing gap (oracle − champion = 3 tasks) cannot be recovered
with TF-IDF signals; it requires operation-aware routing.

## 5. Related Work

- Tool-augmented LLMs (ReAct, Toolformer)
- Retrieval-augmented generation (RAG) for code (CodeRAG, RepoCoder)
- Memory for agents (Generative Agents, MemGPT)
- LoRA fine-tuning for code (CodeAlpaca, WizardCoder)
- Dense code retrieval (CodeBERT, UniXcoder, CodeT5+)

## 6. Limitations

- Single model family (Qwen2.5-Coder 1.5B)
- 28-task frozen benchmark: hard, single-function Python tasks only
- No comparison to larger models on the same benchmark
- No SWE-bench evaluation
- Sampling variance ±2–3 tasks at best-of-3 is non-trivial relative to total tasks

## 7. Next Work

### 7.1 Operation-aware retrieval (immediate)
Dense code retrieval (CodeBERT, UniXcoder) to replace TF-IDF similarity.
Semantic similarity would distinguish algorithm patterns from vocabulary families.
Routing could then use model-specific similarity thresholds per algorithm class.

### 7.2 SWE-bench infrastructure (engineering)
Extend the evaluation harness to repo-level patch generation.
This requires multi-file context, diff application, and test execution in a sandbox.
Does not require new memory or model changes.

### 7.3 Larger benchmark before any public claim
The 28-task frozen benchmark is too small for strong statistical claims.
A 200+ task benchmark across more algorithm families would reduce variance
and enable between-condition effect sizes to be estimated reliably.

## Appendix

- A: Full experiment results tables → `02_main_results_table.md`
- B: Failure analysis → `03_failure_analysis.md`
- C: Claim boundary → `04_claim_boundary.md`
- D: Reproducibility commands → `05_reproducibility_commands.md`
- E: Per-task routing scores (v2.11) → `results/v211_retrieval_routing/per_task_routing.csv`
- F: Benchmark integrity finding (tree_depth_tuple) → `docs/v2.9_memory_repair_split.md`
