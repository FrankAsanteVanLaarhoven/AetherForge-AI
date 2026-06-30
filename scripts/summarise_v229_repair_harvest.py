"""
scripts/summarise_v229_repair_harvest.py — v2.29 repair-harvest summary (committed evidence).

Reads the LOCAL-ONLY harvest aggregate (build_v229_repair_harvest.py) and the v2.28 dataset
aggregate produced WITH the v2.29 traces included, and writes only small curated summaries safe to
commit. Proves the harvest produces genuine broken→fixed repair transitions and that the v2.28
builder now accepts non-zero format-repair and verifier-format candidates. The full generated
traces/dataset are never read into the summary or committed.

Writes results/v229_repair_harvest/: summary.md, failure_types.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v229_repair_harvest.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v229_repair_harvest"
HARVEST = ROOT / "data" / "generated" / "v229" / "harvest_aggregate.json"
V228 = ROOT / "data" / "generated" / "v228" / "dataset_aggregate.json"


def main():
    if not HARVEST.exists():
        print("No harvest aggregate. Run make build-v229-harvest first.")
        sys.exit(1)
    h = json.loads(HARVEST.read_text())
    d = json.loads(V228.read_text()) if V228.exists() else {}
    uc = d.get("use_tag_counts", {})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "failure_types.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["failure_type", "count"])
        w.writeheader()
        for k, v in sorted(h.get("failure_type_distribution", {}).items()):
            w.writerow({"failure_type": k, "count": v})

    fr = uc.get("format_repair_candidate", 0)
    vf = uc.get("verifier_format_candidate", 0)
    guard_ok = h.get("contamination_guard_violations", 1) == 0 and d.get("contamination_guard_violations", 1) == 0
    promote = (h.get("successful_repairs", 0) > 0 and h.get("all_genuine_transitions") and
               guard_ok and fr > 0 and vf > 0)

    s = ["# v2.29 — Genuine Repair Trace Harvest", "",
         "Manufactures genuine, execution-verified, verifier-labelled FORMAT-repair transitions "
         "(candidate fails → structured verifier signal → canonical repair → final passes) by "
         "perturbing ONLY the output format of known-correct tree serializers. Not training; held-out "
         "evaluation untouched. Full traces/dataset are local-only (gitignored); only this summary is "
         "committed.", "",
         "## Harvest", "",
         f"- Records harvested: **{h.get('records_harvested', 0)}**",
         f"- Repair attempts: **{h.get('repair_attempts', 0)}**",
         f"- Successful repairs (candidate≠final, final verified pass): **{h.get('successful_repairs', 0)}**",
         f"- Dropped (ambiguous / non-format / raised before output): "
         f"{h.get('rejected_nonformat_label', 0)}",
         f"- All transitions genuine (candidate ≠ final): **{h.get('all_genuine_transitions')}**", "",
         "### Failure-type distribution", ""]
    for k, v in sorted(h.get("failure_type_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["", "### Representation distribution", ""]
    for k, v in sorted(h.get("representation_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["",
          "## Fed through the v2.28 dataset builder", "",
          f"- Format-repair candidates: **{fr}** (was 0 in v2.28).",
          f"- Verifier-format candidates: **{vf}** (was 0 in v2.28).",
          f"- SFT candidates: {uc.get('sft_candidate', 0)} | preference-pair: "
          f"{uc.get('preference_pair_candidate', 0)}.",
          f"- Dataset accepted/rejected: {d.get('accepted', '?')}/{d.get('rejected', '?')}.", "",
          "### Quality-score buckets (accepted dataset)", ""]
    for k, v in sorted(d.get("quality_score_buckets", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["",
          "## Contamination guard", "",
          f"- Harvest violations: **{h.get('contamination_guard_violations', 0)}**; dataset "
          f"violations: **{d.get('contamination_guard_violations', 0)}** (computed over "
          "name/function/prompt/solution/test overlap vs the 32-task benchmark and v2.26 slice).", "",
          "## Artifact safety", "",
          "- Harvested traces (`data/generated/v229/`) and the regenerated dataset "
          "(`data/generated/v228/`) are local-only and gitignored; only this curated summary is "
          "committed. No outputs, logs, indexes, checkpoints, weights, or generated JSONL committed.",
          "",
          "## Promotion", "",
          ("**PROMOTE** — genuine broken→fixed repair transitions are produced (candidate ≠ final, "
           "final verified passing), the contamination guard has 0 violations, and the v2.28 builder "
           "accepts non-zero format-repair and verifier-format candidates. Suitable as future "
           "format-repair / verifier-format training data." if promote else
           "**HOLD** — repairs degenerate, unverified, guard tripped, or candidates still zero."), "",
          "See `failure_types.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.29 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.29 produces genuine verifier-labelled repair traces (candidate fails → structured "
          "format verifier signal → canonical repair → verified pass) suitable for FUTURE "
          "format-repair or verifier-format training.", "",
          "## Not claimed", "",
          "- No model improvement, no SWE-bench success, no production reliability, no RL training, no "
          "general SOTA, no frontier-level agent performance.",
          "- No model trained; no weights, champion adapters, or memory indexes touched.",
          "- Failures are controlled output-FORMAT perturbations of known-correct serializers; the "
          "underlying algorithm is preserved. The traces demonstrate repair, not new capability.",
          "- Generated traces/dataset are local-only; only curated summaries are committed.", "",
          "## Contamination", "",
          f"- Harvest guard violations = {h.get('contamination_guard_violations', 0)}; held-out "
          "evaluation tasks and solutions are never used as harvest inputs."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")

    print(f"Wrote {OUT_DIR}/summary.md, failure_types.csv, claim_boundary.md")
    print(f"harvested={h.get('records_harvested')} successes={h.get('successful_repairs')} "
          f"format_repair={fr} verifier_format={vf} promote={promote}")


if __name__ == "__main__":
    main()
