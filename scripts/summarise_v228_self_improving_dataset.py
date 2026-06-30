"""
scripts/summarise_v228_self_improving_dataset.py — v2.28 dataset summary (committed evidence).

Reads the LOCAL-ONLY dataset aggregate produced by build_v228_self_improving_dataset.py (and, for
context, the v2.27 trace aggregate) and writes only SMALL curated summaries safe to commit. The
full generated dataset is never read into the summary or committed.

Writes results/v228_self_improving_dataset/: summary.md, distribution.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v228_self_improving_dataset.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v228_self_improving_dataset"
AGG = ROOT / "data" / "generated" / "v228" / "dataset_aggregate.json"
V227_AGG = ROOT / "data" / "generated" / "v227" / "trace_aggregate.json"


def main():
    if not AGG.exists():
        print("No dataset aggregate. Run make build-v228-dataset first.")
        sys.exit(1)
    a = json.loads(AGG.read_text())
    v227 = json.loads(V227_AGG.read_text()) if V227_AGG.exists() else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # distribution.csv
    rows = []
    for k, v in a.get("representation_distribution", {}).items():
        rows.append({"dimension": "representation", "key": k, "count": v})
    for k, v in a.get("task_family_distribution", {}).items():
        rows.append({"dimension": "task_family", "key": k, "count": v})
    for k, v in a.get("split_distribution", {}).items():
        rows.append({"dimension": "split", "key": k, "count": v})
    for k, v in a.get("quality_score_buckets", {}).items():
        rows.append({"dimension": "quality_score", "key": k, "count": v})
    with open(OUT_DIR / "distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "key", "count"])
        w.writeheader(); w.writerows(rows)

    uc = a.get("use_tag_counts", {})
    s = ["# v2.28 — Self-Improving Trace Dataset", "",
         "Normalises the local v2.26 + v2.27 trace reconstructions into one canonical, "
         "contamination-guarded, quality-scored self-improvement dataset schema for LATER SFT / "
         "preference / lightweight scaffold training. Not a training milestone; no weights touched. "
         "The full generated dataset is local-only (gitignored); only this summary is committed.", "",
         "## Counts", "",
         f"- Total records scanned: **{a.get('total_scanned', 0)}** "
         f"(v2.27 primary + v2.26 deduped to the same runs).",
         f"- Accepted: **{a.get('accepted', 0)}** | Rejected: **{a.get('rejected', 0)}**.", "",
         "### Rejection reasons", ""]
    for k, v in sorted(a.get("rejection_reasons", {}).items(), key=lambda x: -x[1]):
        s.append(f"- {k}: {v}")
    s += ["", "### Training-use candidate counts", "",
          f"- SFT candidate: **{uc.get('sft_candidate', 0)}**",
          f"- Preference pair candidate: **{uc.get('preference_pair_candidate', 0)}**",
          f"- Format-repair candidate: **{uc.get('format_repair_candidate', 0)}**",
          f"- Verifier-format candidate: **{uc.get('verifier_format_candidate', 0)}**", "",
          "### Split distribution", ""]
    for k, v in sorted(a.get("split_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["", "### Representation distribution (accepted)", ""]
    for k, v in sorted(a.get("representation_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["", "### Task-family distribution (accepted)", ""]
    for k, v in sorted(a.get("task_family_distribution", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["", "### Quality-score buckets (accepted)", ""]
    for k, v in sorted(a.get("quality_score_buckets", {}).items()):
        s.append(f"- {k}: {v}")
    s += ["",
          "## Contamination guard", "",
          f"- COMPUTED over task-name / function-name / prompt / solution / test-case overlap vs the "
          f"32-task benchmark and the v2.26 slice: violations = "
          f"**{a.get('contamination_guard_violations', 0)}**.", "",
          "## Known limitations", "",
          "- **Format-repair candidates = 0**: in the source runs the model's verifier-repair loop "
          f"fixed 0 genuine failures (v2.27: repair_attempted_fixed=0); the "
          f"{v227.get('envelope_format_failures', 0)} envelope-format failures were algorithm-correct "
          "(tool-call/output ENVELOPE format), and the degenerate repair records (claimed repair, no "
          "code change) are filtered. So this dataset yields SFT positives and some preference pairs "
          "but NOT usable broken→fixed repair pairs yet.",
          "- The corpus is bounded to the tree_serialize representation family (3B-bf16 runs); it is a "
          "substrate, not a broad agentic dataset.",
          "- Preference pairs are cross-record (pass vs fail for the same task/representation), not "
          "within-trajectory.", "",
          "## Promotion", "",
          ("**PROMOTE** — the repo generates a contamination-guarded, quality-scored self-improving "
           "trace dataset locally, with this committed summary proving record counts, filter "
           "decisions, and artifact safety. The substrate supports future SFT and preference training; "
           "repair/verifier-format training needs a richer trace harvest (next milestone)."
           if a.get("accepted", 0) > 0 and a.get("contamination_guard_violations", 0) == 0 else
           "**HOLD** — dataset too incomplete or contamination guard tripped."), "",
          "See `distribution.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.28 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.28 establishes a trace-dataset SUBSTRATE for later self-improving training: a "
          "canonical, contamination-guarded, quality-scored record schema and the source-only tooling "
          "to generate it locally from the v2.26/v2.27 traces.", "",
          "## Not claimed", "",
          "- No model improvement, no SWE-bench success, no production reliability, no RL training, no "
          "general SOTA, no frontier-level agent performance.",
          "- No model was trained; no weights, champion adapters, or memory indexes were touched.",
          "- The generated dataset is local-only; only this curated summary is committed.", "",
          "## Contamination", "",
          f"- Computed guard violations across the accepted+rejected set = "
          f"{a.get('contamination_guard_violations', 0)} (name/function/prompt/solution/test overlap "
          "vs the 32-task benchmark and v2.26 slice)."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")

    print(f"Wrote {OUT_DIR}/summary.md, distribution.csv, claim_boundary.md")
    print(f"accepted={a.get('accepted')} rejected={a.get('rejected')} "
          f"sft={uc.get('sft_candidate')} pref={uc.get('preference_pair_candidate')} "
          f"guard_violations={a.get('contamination_guard_violations')}")


if __name__ == "__main__":
    main()
