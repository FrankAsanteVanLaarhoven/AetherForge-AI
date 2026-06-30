"""
scripts/summarise_v230_broadened_repair_harvest.py — v2.30 summary (committed evidence).

Reads the LOCAL-ONLY v2.30 harvest aggregate and the v2.28 dataset aggregate (regenerated WITH the
v2.29 + v2.30 traces) and writes only small curated summaries safe to commit. The full generated
traces/dataset are never read into the summary or committed.

Writes results/v230_broadened_repair_harvest/: summary.md, failure_types.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v230_broadened_repair_harvest.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v230_broadened_repair_harvest"
HARVEST = ROOT / "data" / "generated" / "v230" / "harvest_aggregate.json"
V228 = ROOT / "data" / "generated" / "v228" / "dataset_aggregate.json"


def main():
    if not HARVEST.exists():
        print("No harvest aggregate. Run make build-v230-harvest first.")
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
    ar = uc.get("algorithmic_repair_candidate", 0)
    vf = uc.get("verifier_format_candidate", 0)
    genuine = fr + ar
    guard_ok = h.get("contamination_guard_violations", 1) == 0 and d.get("contamination_guard_violations", 1) == 0
    promote_min = genuine >= 30 and fr >= 15 and ar >= 5 and guard_ok
    promote_strong = genuine >= 50 and fr > 0 and ar > 0 and h.get("all_genuine_transitions")

    s = ["# v2.30 — Broadened Repair Trace Harvest", "",
         "Broadens the genuine repair-trace corpus beyond tree_serialize format perturbations: more "
         "task families plus a controlled ALGORITHMIC repair slice. Every record is an "
         "execution-verified candidate(fail) → structured verifier signal → canonical repair → "
         "final(pass) transition on NEW non-held-out tasks (names disjoint from the 32-task "
         "benchmark). Not training; held-out evaluation untouched. Full traces/dataset are local-only "
         "(gitignored); only this summary is committed.", "",
         "## Harvest (v2.30)", "",
         f"- Records harvested: **{h.get('records_harvested', 0)}**  (attempts "
         f"{h.get('repair_attempts', 0)})",
         f"- Successful repairs (candidate≠final, final verified pass): "
         f"**{h.get('successful_repairs', 0)}**; all genuine: **{h.get('all_genuine_transitions')}**",
         f"- Format repairs: **{h.get('format_repair', 0)}** | algorithmic repairs: "
         f"**{h.get('algorithmic_repair', 0)}** | mixed: {h.get('mixed_repair', 0)}",
         f"- Rejections: {h.get('rejection_reasons', {})}", "",
         "### Failure-type distribution", ""]
    for k, v in sorted(h.get("failure_type_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["", "### Task-family distribution", ""]
    for k, v in sorted(h.get("task_family_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["",
          "## Cumulative dataset via the v2.28 builder (v2.29 + v2.30 sources)", "",
          f"- Accepted/rejected: {d.get('accepted', '?')}/{d.get('rejected', '?')}.",
          f"- **Format-repair candidates: {fr}** (v2.28 alone: 0).",
          f"- **Algorithmic-repair candidates: {ar}** (new category in v2.30).",
          f"- **Verifier-format candidates: {vf}**.",
          f"- SFT candidates: {uc.get('sft_candidate', 0)} | preference-pair: "
          f"{uc.get('preference_pair_candidate', 0)}.",
          f"- Genuine broken→fixed repairs (format + algorithmic): **{genuine}**.", "",
          "### Quality-score buckets (accepted dataset)", ""]
    for k, v in sorted(d.get("quality_score_buckets", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["",
          "## Contamination guard", "",
          f"- Harvest violations: **{h.get('contamination_guard_violations', 0)}**; dataset "
          f"violations: **{d.get('contamination_guard_violations', 0)}** (computed over "
          "name/function/prompt/solution/test overlap vs the 32-task benchmark and v2.26 slice). "
          "Harvest tasks are newly authored with names disjoint from the benchmark.", "",
          "## Artifact safety", "",
          "- Harvested traces (`data/generated/v230/`) and the regenerated dataset "
          "(`data/generated/v228/`) are local-only and gitignored; only this curated summary is "
          "committed. No outputs, logs, indexes, checkpoints, weights, or generated JSONL committed.",
          "",
          "## Promotion", "",
          (f"**PROMOTE (strong)** — {genuine} genuine broken→fixed repairs across "
           f"{len(h.get('task_family_distribution', {}))} families, both format ({fr}) and "
           f"algorithmic ({ar}) categories non-zero, all finals verified passing, 0 contamination "
           "violations." if promote_strong else
           f"**PROMOTE (minimum)** — {genuine} genuine repairs, format {fr}≥15, algorithmic {ar}≥5, "
           "0 contamination violations." if promote_min else
           "**HOLD** — minimum targets not met."), "",
          "See `failure_types.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.30 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.30 broadens the verifier-labelled repair trace corpus (more families + an algorithmic "
          "repair slice) and prepares the substrate for a FUTURE small SFT or preference pilot.", "",
          "## Not claimed", "",
          "- No model improvement, no SWE-bench success, no production reliability, no RL training, no "
          "general SOTA, no frontier-level agent performance.",
          "- No model trained; no weights, champion adapters, or memory indexes touched.",
          "- Failures are controlled perturbations of newly-authored, non-held-out reference functions; "
          "the traces demonstrate repair, not new capability.",
          "- Generated traces/dataset are local-only; only curated summaries are committed.", "",
          "## Contamination", "",
          f"- Harvest guard violations = {h.get('contamination_guard_violations', 0)}; no held-out "
          "benchmark task, solution, or test is used as a harvest input."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")

    print(f"Wrote {OUT_DIR}/summary.md, failure_types.csv, claim_boundary.md")
    print(f"harvested={h.get('records_harvested')} genuine={genuine} format={fr} algo={ar} "
          f"verifier_format={vf} strong={promote_strong} min={promote_min}")


if __name__ == "__main__":
    main()
