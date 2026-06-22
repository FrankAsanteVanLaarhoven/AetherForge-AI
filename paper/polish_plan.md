# Paper Polish Plan

**Source:** `paper/aetherforge_memory_augmented_code_agent_draft.md`  
**Target:** Submission-ready manuscript  
**Status:** Draft complete (v2.13). This plan identifies every required change before the paper is ready for submission or public release.

---

## Priority Order

```
1. Abstract (replace with abstract_v2.md — done)
2. Introduction (tighten claim, add retrieval-noise framing early)
3. Results §5 (split positive from negative more sharply)
4. Failure Analysis §6 (promote to full narrative section)
5. Limitations §7 (add benchmark size caveat explicitly)
6. Tables (move all to paper/tables/ and reference from body)
7. Figures (render from specs in paper/figures/)
8. Appendix (move reproducibility commands + per-task tables)
9. Final claim-boundary pass across the whole document
```

---

## Section-by-Section Changes

### Abstract

**Action:** Replace with `paper/abstract_v2.md`.  
**Reason:** Current abstract is 220 words with adequate content but needs one structural
fix: the sentence "we identify three retrieval failure modes" should appear before the
oracle ceiling sentence, not after. The flow is: system → positive finding → five
negatives → root cause → clean result.

---

### §1 Introduction

**Current state:** Correctly frames the local-model / offline-retrieval problem.
Good motivation. Needs one change.

**Actions:**
- Add a sentence in paragraph 2 naming retrieval noise as the mechanism at the core of
  all five negative results. Currently the intro promises "we identify failure modes" but
  does not prime the reader on what kind of failure. Adding "specifically, bag-of-words
  similarity conflates vocabulary co-occurrence with algorithmic relevance" in the
  intro makes §6 feel like a confirmation rather than a surprise.
- Remove or weaken: "This allows the model to benefit from previously solved patterns
  without retraining." True, but it understates the qualification — memory helps when
  retrieval is accurate and hurts when it is not. Replace with a more conditional
  framing.
- Contributions list: move to end of intro (currently buried in §1.3-equivalent prose).
  Three bullets: (1) load-bearing measurement, (2) failure mode taxonomy, (3) oracle
  ceiling establishing the routing bottleneck.

---

### §2 Problem

**Current state:** Good and tight. No major changes needed.

**Action:** One word change — "two competing pressures" is correct but abstract. Add a
one-sentence concrete example: "A model that retrieves the `merge_sorted` pattern when
asked to solve `merge_intervals` produces code that sorts two lists rather than merging
overlapping intervals — a failure invisible to the retriever." This makes the problem
tangible before §3.

---

### §3 System Overview

**Current state:** Accurate and complete.

**Actions:**
- Add one sentence explicitly distinguishing the clean champion index from the repair
  diagnostic index. Currently the distinction is in a table but not stated in prose.
  Add: "The champion index (99 records) is the only configuration evaluated on clean
  held-out benchmarks; the repair index (103 records) is evaluated diagnostically."
- Move the VRAM table to a footnote or appendix — it adds bulk without serving the
  scientific argument.

---

### §4 Experimental Setup

**Current state:** Correct. One gap.

**Action:** Add explicit contamination rule: "Any benchmark that receives repair records
targeting known failures is immediately reclassified from clean to diagnostic; results
on that benchmark are no longer reported as the clean held-out champion."  
This sentence pre-emptively answers the most likely reviewer question about the 96.4%
result.

---

### §5 Results

**Current state:** 5 subsections covering the positive finding and 4 negatives.
The structure is correct. Three targeted changes:

**§5.1 (Memory is load-bearing):**
- The word "decorative" is unnecessarily casual. Replace with "passive" or
  "non-essential." "Memory retrieval is not passive — it is required for 6 tasks..."
- Add a sentence: "The six tasks that fail entirely without memory cover diverse patterns
  (graph traversal, caching, topo-sort, IP validation), suggesting the lift is broad
  rather than concentrated in one algorithm category."

**§5.3 (Repair memory diagnostic):**
- The label "Diagnostic only" must appear in the section heading, not just in the table
  caption. Change `### 5.3 Targeted Repair Memory Fixes Known Failures (Diagnostic)` to
  make the word "diagnostic" prominent, and open the section with: "This result is
  diagnostic, not a clean held-out champion. The frozen benchmark is non-independent for
  this evaluation because repair records target the same known failures."
- Move the 96.4% number to a display-math or table row that is visually distinct from
  the 82.1% clean champion number. A reviewer should not be able to mistake one for the
  other at a glance.

**§5.4 (Repair memory generalisation):**
- The sentence "Net task flip is −2" is confusing. Correct count: +2 gained, −4 lost.
  The net is −2 but should be expressed as: "Repair gains 2 tasks and loses 4, for a net
  change of −2 tasks (−6.3 pp)."
- Add a sentence explicitly attributing the per-family pattern: "The interval_merge
  family is the only one where repair helps (+16.7 pp), consistent with those repair
  records targeting interval-specific patterns. The three regressing families all share
  vocabulary with repair records not designed for them."

**§5.5 (Routing):**
- Add: "The confidence router routes 20 of 32 tasks to the repair index — indicating the
  threshold (top-1 ≥ 0.35, margin ≥ 0.05) is too permissive, not too strict."
- Oracle breakdown sentence needs to be prominent: "Only 3 tasks benefit from repair
  routing under any scheme; 9 tasks fail regardless of routing choice." This sentence
  should appear before the table, not after.

---

### §6 Failure Analysis

**Current state:** Good but underweighted. Currently placed after Limitations. Should
be elevated — this is the paper's second-strongest contribution after the load-bearing
finding.

**Actions:**
- **Promote §6 to immediately after §5.** The failure taxonomy is the explanation for
  all five negative results. Putting it after Limitations buries it.
- Add a unifying paragraph before the three types: "All three failure classes arise from
  the same root cause: TF-IDF similarity is computed over token frequency, not over
  algorithm structure. A retrieval signal that encodes what a function does rather than
  what words it uses would resolve all three types."
- §6.2 (Benchmark integrity): Fine. Add one sentence at the start: "This finding
  illustrates the value of forensic analysis beyond aggregate scores — a task that
  consistently fails despite correct implementations reveals a benchmark defect, not a
  model defect."
- §6.3 (Sampling variance): Add explicit implication sentence: "Under this variance
  budget, a difference of fewer than 3 tasks on the 32-task benchmark cannot be
  attributed to the experimental condition rather than sampling noise."

---

### §7 Limitations

**Current state:** Four bullets. All accurate. Two additions:

- After the "Scale" bullet: add that the 28-task benchmark was constructed to be hard
  (single-function, assertion-verified, covering diverse categories), so the absolute
  numbers are not comparable to self-reported scores on easier benchmarks.
- Add a fifth limitation: "All results use a single model family (Qwen2.5-Coder-1.5B).
  Whether the memory lift or the retrieval failure modes transfer to other architectures
  or sizes is untested."

---

### §8 Reproducibility

**Current state:** Fine. One change.

**Action:** Add at the top: "All results in this paper were obtained on a single RTX
4080 Super (16 GB VRAM). No cloud inference or external API was used." This is important
for reproducibility claims and distinguishes the work from API-dependent systems.

---

### §9 Future Work

**Current state:** Good. One addition.

**Action:** Add to the dense-retrieval paragraph: "The oracle ceiling (23/32 = 71.9%)
provides a concrete target: a perfect retrieval signal would recover 3 additional tasks.
Dense retrieval does not guarantee reaching this ceiling, but it removes the vocabulary-
overlap mechanism that currently blocks it."

---

### §10 Conclusion

**Current state:** Strong. No required changes. Optional tightening:

- The sentence "The clean champion — 23/28 = 82.1% on the frozen benchmark, fully
  reproducible — is the stable result." is good but can be strengthened: "...is the
  stable, reproducible result for this arc, and establishes a concrete baseline for
  future retrieval-improvement experiments."

---

## Tables

**Current placement:** Inline in results sections.  
**Target:** All tables in `paper/tables/main_results_table.md`, referenced from body.

| Table | Body reference | File target |
|---|---|---|
| Core system performance | §5.1 | Table 1 |
| Retraining experiments | §5.2 | Table 2 |
| Memory ablation | §5.2 | Table 3 |
| Repair memory (v2.9) | §5.3 | Table 4 |
| Per-family v2.10 | §5.4 | Table 5 |
| Routing audit v2.11 | §5.5 | Table 6 |

**Action:** Replace inline table Markdown with `[Table N]` cross-references and move
full tables to an appendix section or `paper/tables/`.

---

## Figures

**Status:** Specs written in `paper/figures/`. Not yet rendered.

| Figure | Spec file | Priority |
|---|---|---|
| Experiment timeline bar chart | `experiment_timeline_spec.md` | High |
| Retrieval failure taxonomy | `retrieval_failure_taxonomy_spec.md` | High |

**Action:** Render from specs using matplotlib (scripts exist in notebook). Save as
`paper/figures/fig_experiment_timeline.png` and `paper/figures/fig_failure_taxonomy.png`.
Then reference from §5 and §6 respectively.

---

## Appendix

Move to appendix (does not belong in main body):

- Full Makefile reproducibility commands (currently in §8) — keep a one-line summary in
  body, move full listing to appendix
- Per-task routing decisions table (currently in v2.11 results doc)
- VRAM budget table (currently in §3)

---

## Claims Requiring Weakening

| Current phrasing | Required change |
|---|---|
| "memory retrieval is not decorative" | → "memory retrieval is not passive" |
| "repair memory fixes known failures" | → "repair memory fixes known failures *diagnostically*" (add qualifier) |
| "the model must produce its own tool call" | Already correct — keep |
| Any implied claim that 96.4% is reproducible on unseen tasks | Must not appear |
| Any claim about "production-grade" | Already absent — keep absent |

---

## Claims That Must Remain

| Claim | Evidence |
|---|---|
| Memory lift +17.8 pp (clean, controlled) | v2.7 ablation, Table 3 |
| All retraining configurations regress | v2.5/v2.6, Table 2 |
| Repair fails on 32-task clean benchmark | v2.10, Table 5 |
| No routing strategy beats champion | v2.11, Table 6 |
| Oracle ceiling 71.9%, 9 tasks unreachable | v2.11 oracle analysis |
| Three retrieval failure modes with shared root cause | §6 taxonomy |
| Benchmark defect in tree_depth_tuple | §6.2 |

---

## What Must Not Appear

- 27/28 = 96.4% labelled as "champion" or "clean" anywhere in the paper
- SWE-bench claims
- Production or deployment claims
- "The system generalises to..." without explicit scope qualification
- Comparison statements to specific models without co-evaluation
