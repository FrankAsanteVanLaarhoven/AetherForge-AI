# v2.29 — Genuine Repair Trace Harvest

Manufactures genuine, execution-verified, verifier-labelled FORMAT-repair transitions (candidate fails → structured verifier signal → canonical repair → final passes) by perturbing ONLY the output format of known-correct tree serializers. Not training; held-out evaluation untouched. Full traces/dataset are local-only (gitignored); only this summary is committed.

## Harvest

- Records harvested: **9**
- Repair attempts: **21**
- Successful repairs (candidate≠final, final verified pass): **9**
- Dropped (ambiguous / non-format / raised before output): 12
- All transitions genuine (candidate ≠ final): **True**

### Failure-type distribution

- extra_null_marker: 2
- format_error: 1
- missing_null_marker: 1
- separator_error: 1
- type_error: 4

### Representation distribution

- exact_string: 4
- nested_list: 3
- token_list: 2

## Fed through the v2.28 dataset builder

- Format-repair candidates: **9** (was 0 in v2.28).
- Verifier-format candidates: **9** (was 0 in v2.28).
- SFT candidates: 35 | preference-pair: 8.
- Dataset accepted/rejected: 38/43.

### Quality-score buckets (accepted dataset)

- 0.4-0.6: 3
- 0.6-0.8: 22
- 0.8-1.0: 13

## Contamination guard

- Harvest violations: **0**; dataset violations: **0** (computed over name/function/prompt/solution/test overlap vs the 32-task benchmark and v2.26 slice).

## Artifact safety

- Harvested traces (`data/generated/v229/`) and the regenerated dataset (`data/generated/v228/`) are local-only and gitignored; only this curated summary is committed. No outputs, logs, indexes, checkpoints, weights, or generated JSONL committed.

## Promotion

**PROMOTE** — genuine broken→fixed repair transitions are produced (candidate ≠ final, final verified passing), the contamination guard has 0 violations, and the v2.28 builder accepts non-zero format-repair and verifier-format candidates. Suitable as future format-repair / verifier-format training data.

See `failure_types.csv`, `claim_boundary.md`.
