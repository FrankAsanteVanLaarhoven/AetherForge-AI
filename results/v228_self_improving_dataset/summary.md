# v2.28 — Self-Improving Trace Dataset

Normalises the local v2.26 + v2.27 trace reconstructions into one canonical, contamination-guarded, quality-scored self-improvement dataset schema for LATER SFT / preference / lightweight scaffold training. Not a training milestone; no weights touched. The full generated dataset is local-only (gitignored); only this summary is committed.

## Counts

- Total records scanned: **72** (v2.27 primary + v2.26 deduped to the same runs).
- Accepted: **29** | Rejected: **43**.

### Rejection reasons

- superseded_by_higher_priority_source: 36
- claimed_repair_without_change: 7

### Training-use candidate counts

- SFT candidate: **26**
- Preference pair candidate: **7**
- Format-repair candidate: **0**
- Verifier-format candidate: **0**

### Split distribution

- preference_candidate: 3
- rejected: 43
- train_candidate: 26

### Representation distribution (accepted)

- exact_string: 8
- json: 9
- nested_list: 7
- token_list: 5

### Task-family distribution (accepted)

- tree_serialize_repr: 29

### Quality-score buckets (accepted)

- 0.4-0.6: 3
- 0.6-0.8: 22
- 0.8-1.0: 4

## Contamination guard

- COMPUTED over task-name / function-name / prompt / solution / test-case overlap vs the 32-task benchmark and the v2.26 slice: violations = **0**.

## Known limitations

- **Format-repair candidates = 0**: in the source runs the model's verifier-repair loop fixed 0 genuine failures (v2.27: repair_attempted_fixed=0); the 15 envelope-format failures were algorithm-correct (tool-call/output ENVELOPE format), and the degenerate repair records (claimed repair, no code change) are filtered. So this dataset yields SFT positives and some preference pairs but NOT usable broken→fixed repair pairs yet.
- The corpus is bounded to the tree_serialize representation family (3B-bf16 runs); it is a substrate, not a broad agentic dataset.
- Preference pairs are cross-record (pass vs fail for the same task/representation), not within-trajectory.

## Promotion

**PROMOTE** — the repo generates a contamination-guarded, quality-scored self-improving trace dataset locally, with this committed summary proving record counts, filter decisions, and artifact safety. The substrate supports future SFT and preference training; repair/verifier-format training needs a richer trace harvest (next milestone).

See `distribution.csv`, `claim_boundary.md`.
