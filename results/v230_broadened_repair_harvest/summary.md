# v2.30 — Broadened Repair Trace Harvest

Broadens the genuine repair-trace corpus beyond tree_serialize format perturbations: more task families plus a controlled ALGORITHMIC repair slice. Every record is an execution-verified candidate(fail) → structured verifier signal → canonical repair → final(pass) transition on NEW non-held-out tasks (names disjoint from the 32-task benchmark). Not training; held-out evaluation untouched. Full traces/dataset are local-only (gitignored); only this summary is committed.

## Harvest (v2.30)

- Records harvested: **41**  (attempts 43)
- Successful repairs (candidate≠final, final verified pass): **41**; all genuine: **True**
- Format repairs: **19** | algorithmic repairs: **22** | mixed: 0
- Rejections: {'perturbation_did_not_break': 2}

### Failure-type distribution

- algorithmic_error: 22
- format_error: 8
- separator_error: 8
- type_error: 3

### Task-family distribution

- arithmetic: 8
- container_format: 6
- json_format: 1
- kv_format: 4
- list_format: 6
- recursion: 6
- scan: 6
- sequence: 2
- string_format: 2

## Cumulative dataset via the v2.28 builder (v2.29 + v2.30 sources)

- Accepted/rejected: 79/43.
- **Format-repair candidates: 28** (v2.28 alone: 0).
- **Algorithmic-repair candidates: 22** (new category in v2.30).
- **Verifier-format candidates: 28**.
- SFT candidates: 76 | preference-pair: 8.
- Genuine broken→fixed repairs (format + algorithmic): **50**.

### Quality-score buckets (accepted dataset)

- 0.4-0.6: 3
- 0.6-0.8: 22
- 0.8-1.0: 54

## Contamination guard

- Harvest violations: **0**; dataset violations: **0** (computed over name/function/prompt/solution/test overlap vs the 32-task benchmark and v2.26 slice). Harvest tasks are newly authored with names disjoint from the benchmark.

## Artifact safety

- Harvested traces (`data/generated/v230/`) and the regenerated dataset (`data/generated/v228/`) are local-only and gitignored; only this curated summary is committed. No outputs, logs, indexes, checkpoints, weights, or generated JSONL committed.

## Promotion

**PROMOTE (strong)** — 50 genuine broken→fixed repairs across 9 families, both format (28) and algorithmic (22) categories non-zero, all finals verified passing, 0 contamination violations.

See `failure_types.csv`, `claim_boundary.md`.
