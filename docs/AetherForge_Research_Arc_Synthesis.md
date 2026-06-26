# AetherForge Research Arc Synthesis (v2.17 → v2.23)

A single, ablation-backed account of the retrieval-and-reasoning arc on the clean 32-task
generalisation benchmark. Every quantitative claim below is reproduced by a committed
`results/` directory and a `make` target. No model weights, dense indexes, or eval outputs
are committed; this document and the small evidence summaries are the durable record.

## 1. Goal

Map, on the frozen 32-task benchmark, which failures are **fixable** by retrieval/inference
interventions and which are **capability-bound** (requiring weight-level change) — and
attribute every improvement to a specific cause via controlled ablations, never to a
confounded mixture.

## 2. The stabilised baseline (and a correction)

The protected memory baseline (`memory/index_adapted`, 99 verified records) was historically
described as "TF-IDF". v2.18 Phase A established by inspection (vocab_size=0, dim=384, L2-norm=1)
that it is in fact **code-aware dense retrieval** — local `nreimers/MiniLM-L6-H384-uncased`
fine-tuned on code_search_net. The TF-IDF fallback in `memory/embed.py` has never been active.
All prior "TF-IDF" wording was corrected.

Stabilised baseline over three runs: **16.3/32 = 51.0%** (17, 15, 17), range 15–17, std 0.94.
Structure: 11 stable-pass, 11 stable-fail, 10 flip tasks. The historical 20/32 (v2.10) is a
high-tail flip-task draw, not degradation. **Promotion gate (used throughout): mean > 18.3/32
over three runs (baseline + a 2-task noise floor), and — for a real win — at least one
stable-fail → stable-pass conversion without broad family regression.**

## 3. The four levers and their attributed conclusions

| Lever | What it fixes | Ablation proof | Status |
|---|---|---|---|
| Coverage (family-relevant memory) | interval failures | v2.19c: `interval_union` 0→3/3 once interval records exist | resolved (positive) |
| Planning structure (execution-plan prompt) | `tree_width` | v2.21b: survives removal of the worked example (9/9) | resolved (positive) |
| Verifier **signal format** (structured VERIFIER block) | broad aggregate lift to 22.0 | v2.22b: raw stderr collapses to baseline (16.7 vs 22.0) | resolved (positive) |
| Targeted fine-tuning (contamination-guarded LoRA) | nothing, on the 3 hard tasks | v2.23: no conversion, and no regression | resolved (negative) |

## 4. Milestone-by-milestone

- **v2.17 — generic dense pilot.** Generic `all-MiniLM-L6-v2` dense tied the baseline (17/32);
  hybrid regressed. Null result for generic MiniLM — not evidence against dense retrieval.
- **v2.18 Phase A — baseline correction + stabilisation** (above). Phase B — **UniXcoder-base
  (768d) code-dense**: 18.3/32 (15,20,20), hybrid 17.7. Directional, on the gate, **not
  promoted**. Encoder capacity alone did not move the gate.
- **v2.19 — structured memory + multi-view rerank** (encoder fixed): dense 18.0, hybrid 19.3.
  **Provisional** — the gate was met on the mean but with zero stable conversions, high
  variance, and a retrieval trace showing the surfaced records were family-irrelevant. Finding:
  the bottleneck is memory **coverage**, not format.
- **v2.19b — family-targeted memory.** Adding 16 verified same-family-different-task records →
  19.3/32 (19,20,19), low variance, **`interval_intersection` 0→3/3** (coverage-attributable).
  First **promoted candidate**.
- **v2.19c — confirmation + targeted coverage.** 10 identical-setup seeds → mean **19.8/32**
  (8/10 above gate, variance real). Targeted coverage **converts `interval_union` 0→3/3** but no
  tree stable-fail moves despite relevant retrieval → **split**: interval is coverage-bound,
  tree is reasoning/control-bound.
- **v2.21 — execution-plan curriculum.** A PLAN→code→test→repair prompt **converts `tree_width`
  0→3/3 (6/6)** — the first tree conversion. Guard metrics show 100% plan adherence, so the
  three remaining tree tasks fail *despite* correct planning. **Candidate.**
- **v2.21b — planning ablation.** The same prompt **without the worked example** still converts
  `tree_width` 9/9 → the conversion is caused by the plan **structure**, not the example.
- **v2.22 — verifier-guided repair.** A precise VERIFIER signal + bounded repair budget lifts
  **full-32 to 22.0/32 (23,23,20)**, 61/105 trajectories repaired to PASS, no regressions — the
  highest aggregate in the arc — but **none of the 3 hard tasks stably converts.**
- **v2.22b — repair-signal ablation.** Same budget/no-repeat/diagnostic asserts, **raw stderr
  instead of the structured block** → 16.7/32 (baseline). The 22.0 lift is **attributed to the
  structured signal format**, not the repair discipline.
- **v2.23 — targeted capability adapter.** A small fresh LoRA (50 steps, lr 1e-5) on the merged
  champion, trained on 49 contamination-guarded same-family-different-task tree repair traces.
  **No conversion** (serialize 1/3, from_list 1/3, max_path_sum 0/3) — *and* **no regression**
  (full-32 22.7, no hard regressions), unlike every prior retrain.

## 5. The failure taxonomy (the arc's main product)

| Failure | Root cause | Fixed by | Evidence |
|---|---|---|---|
| interval (e.g. `interval_union`) | missing family-relevant memory | coverage | v2.19b/c |
| `tree_width` | recursive control, not coverage | planning structure | v2.21 + v2.21b |
| broad flip-task underperformance | unactionable failure feedback | structured verifier signal | v2.22 + v2.22b |
| `tree_serialize`, `tree_from_list`, `tree_max_path_sum` | capability | nothing tried | resist all four levers |

The three hard tree tasks (recursive string serialization, balanced reconstruction, any-path
DP) are a **genuine residual capability wall**: they survive coverage, planning, verifier-guided
repair, and a contamination-guarded targeted fine-tune.

## 6. The frozen champion (unchanged throughout)

The trained model — Qwen2.5-Coder-1.5B-Instruct + a 300-step LoRA, merged
(`outputs/qwen15b_v27_champion_merged`) — and its protected memory index (`memory/index_adapted`)
are **frozen and untouched** by this entire arc. Reference: **23/28 = 82.1%** on the frozen
held-out benchmark; **16.3/32** on this clean generalisation benchmark. Every promotion in the
arc (v2.19b coverage, v2.21 planning, v2.22 verifier) is a **retrieval/prompt/repair
configuration** candidate or an **additive adapter** — never a champion replacement.

## 7. Methodological findings (independent of the scores)

- **Ablation before building.** Each positive result was confirmed by a single-variable control
  (v2.21b worked-example, v2.22b raw-stderr) before being built on. The v2.22 aggregate lift was
  *not* claimed until v2.22b attributed it.
- **Contamination discipline.** Every memory/training addition used same-family-*different*-task,
  execution-verified, name-disjoint records, with a runnable guard. The guard caught a real
  substring leak (`subtree_sum` ⊃ `tree_sum`) in v2.23. The hard tasks stayed evaluation-only.
- **Controlled fine-tuning is safe here.** v2.23 is the first retrain in the project that did
  **not** regress (full-32 22.7 vs 22.0). The prior failures (v2.5 53.6%, v2.6 57.1%, Option A
  64.3%) came from broad data-mixture/high-LR retrains that overwrote the champion's
  generalisation; a small, low-LR, separate-adapter-on-merged-champion pilot avoids that.
- **Noise discipline.** A 3-run snapshot can mislead (v2.19b's low-variance 19/20/19 became
  mean 19.8 std 2.14 over 10 seeds). Means over ≥3 runs, a 2-task noise floor, and
  conversion-based (not mean-based) promotion were used throughout.

## 8. Honest limitations / what remains open

- **Not claimed:** that the three hard tasks are unsolvable at 1.5B; that more targeted data or
  more steps would be ineffective (untested under the same guarded protocol); any SWE-bench,
  production-reliability, or frontier-superiority claim.
- **Variance.** At n=32, best-of-3, single-task differences of ±1/3 are flip-level noise;
  promotion was therefore gated on stable (3/3) conversions plus regression checks.
- **Scope.** All results are bounded to this 32-task benchmark and its families.
- **Deferred levers.** The 4-bit / GGUF high-capacity embedding backend (`nomic-embed-code`) was
  repeatedly deferred and never run — the residual failures are reasoning/capability-bound, where
  a heavier embedder is not the indicated fix.

## 9. Recommendations for future work (if pursued)

1. **Scale the v2.23 pilot under the same regression gate** (more contamination-guarded targeted
   data and/or steps) to distinguish "needs more data" from a true capability ceiling — this is
   the single most informative open question, and v2.23 showed it can be done safely.
2. **Operation-aware / shorter memory records** for retrieval (the memory-record-format direction
   flagged since v2.19) — orthogonal to capability.
3. **Keep the structured verifier in the default inference config** — it is the arc's largest
   attributed, regression-free improvement (16.3 → 22.0).

## 10. Reproducibility

Per-milestone evidence and `make` targets:
`results/v218_retrieval_stability/`, `results/v219_structured_memory/`,
`results/v219b_family_memory/`, `results/v219c_confirmation_coverage/`,
`results/v221_reasoning_curriculum/`, `results/v221b_tree_width_ablation/`,
`results/v222_verifier_guided_repair/`, `results/v222b_repair_signal_ablation/`,
`results/v223_tree_capability_adapter/`. Branches `feature/v219*`, `feature/v221*`,
`feature/v222*`, `feature/v223*` are pushed to origin; none merged to `main`.

## 11. Consolidated claim boundary

- **Proven:** interval failures are coverage-bound; `tree_width` is planning/control-bound and
  the conversion is example-independent; the broad aggregate lift to 22.0 is caused by the
  structured verifier signal format (raw-stderr control = baseline); the three hard tree tasks
  resist coverage, planning, verifier-repair, and a contamination-guarded targeted fine-tune; a
  small controlled adapter does not regress the benchmark.
- **Not claimed:** SWE-bench success; production reliability; general tree reasoning solved;
  frontier-model superiority; broad SOTA; that the hard tasks are unsolvable or that scaling
  would not help.

## 12. Attribution

Repository identity is only Frank Asante Van Laarhoven <frankleroyvan@gmail.com>. No
AI/tool/vendor attribution of any kind appears in commits, code, or docs. External model
identifiers (e.g. Qwen, MiniLM, UniXcoder) appear only where technically required for loading
and evaluation configuration.
