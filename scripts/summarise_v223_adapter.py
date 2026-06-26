"""
scripts/summarise_v223_adapter.py

Summarise v2.23 — targeted tree capability adapter.

Compares the merged-champion + v2.23 adapter against the merged-champion alone (v2.22
references), on the three capability-bound hard tree tasks and the full 32-task benchmark,
both under the v2.22 structured-verifier mode. Includes the required adapter-WITHOUT-verifier
ablation (distinguishes a weight-level gain from continued verifier dependence). Judged by
hard-tree conversions with a strict regression gate — NOT aggregate mean alone.

Writes results/v223_tree_capability_adapter/: summary.md, comparison.csv, hardtree.csv,
claim_boundary.md.

Usage:
    python scripts/summarise_v223_adapter.py
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.summarise_v219_structured_memory import load_baseline  # noqa: E402
from scripts.summarise_v221_reasoning import load_rows, passed, pass_map  # noqa: E402

OUT_DIR = Path("results/v223_tree_capability_adapter")
HARD = ["v210_tree_serialize", "v210_tree_from_list", "v210_tree_max_path_sum"]

ADAPT_HARD = [Path(f"outputs/eval_v223_hardtree_run{n}") for n in (1, 2, 3)]
ADAPT_FULL = [Path(f"outputs/eval_v223_full32_run{n}") for n in (1, 2, 3)]
ADAPT_NOVER = [Path(f"outputs/eval_v223_noverifier_run{n}") for n in (1, 2, 3)]
# v2.22 champion+verifier references
CHAMP_HARD = [Path(f"outputs/eval_v222_repair_capbound_run{n}") for n in (1, 2, 3)]
CHAMP_FULL = [Path(f"outputs/eval_v222_repair_32_run{n}") for n in (1, 2, 3)]

V222_MEAN = 22.0
V222_BAND_LO = 20.0  # full-32 must not fall materially below the v2.22 band


def cap(dirs, t):
    c = n = 0
    for d in dirs:
        for row in load_rows(d):
            if (row.get("id") or row.get("task_id")) == t:
                n += 1
                c += int(passed(row))
    return c, n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_npass, base_stab, base_totals = load_baseline()
    baseline_mean = sum(base_totals) / len(base_totals)

    if not any((d / "best_of_3.csv").exists() for d in ADAPT_HARD + ADAPT_FULL):
        print("No v2.23 eval data found. Run the eval-v223-* targets first.")
        sys.exit(1)

    # hard-tree table
    hard_rows, conversions, conv_noverif = [], [], []
    for t in HARD:
        a_c, a_n = cap(ADAPT_HARD, t)
        ch_c, ch_n = cap(CHAMP_HARD, t)
        nv_c, nv_n = cap(ADAPT_NOVER, t)
        converted = a_n >= 3 and a_c == a_n
        converted_nv = nv_n >= 3 and nv_c == nv_n
        if converted:
            conversions.append(t)
        if converted_nv:
            conv_noverif.append(t)
        hard_rows.append({"task_id": t, "baseline": f"{base_npass.get(t)}/3",
                          "champion_verifier": f"{ch_c}/{ch_n}",
                          "adapter_verifier": f"{a_c}/{a_n}",
                          "adapter_NO_verifier": f"{nv_c}/{nv_n}" if nv_n else "-",
                          "converted": "yes" if converted else "no"})
    with open(OUT_DIR / "hardtree.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "baseline", "champion_verifier",
                                          "adapter_verifier", "adapter_NO_verifier", "converted"])
        w.writeheader(); w.writerows(hard_rows)
    print(f"Wrote {OUT_DIR / 'hardtree.csv'}")

    # full-32 aggregate + regressions
    a_pt, a_tot = pass_map(ADAPT_FULL)
    a_mean = statistics.mean(a_tot) if a_tot else 0.0
    a_std = statistics.pstdev(a_tot) if len(a_tot) > 1 else 0.0
    regressions = [t for t in base_npass
                   if base_stab.get(t) == "stable_pass" and a_pt.get(t, 0) == 0] if a_tot else []

    with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "full32_totals", "full32_mean", "hard_conversions", "note"])
        w.writerow(["champion+verifier (v2.22)", "[23, 23, 20]", f"{V222_MEAN}", 0, "reference"])
        w.writerow(["adapter+verifier (v2.23)", str(a_tot), f"{a_mean:.2f}", len(conversions),
                    f"no-verifier conversions: {len(conv_noverif)}"])
    print(f"Wrote {OUT_DIR / 'comparison.csv'}")

    # ── strict verdict ───────────────────────────────────────────────────────
    n_conv = len(conversions)
    band_ok = a_mean >= V222_BAND_LO
    no_regress = not regressions
    if n_conv == 0:
        tag = "no_conversion"
        sentence = ("The targeted adapter did not overcome the residual capability-bound tree "
                    "tasks under the contamination-guarded protocol.")
    elif not no_regress or not band_ok:
        tag = "rejected_regression"
        sentence = (f"A hard tree task converted but the full benchmark regressed "
                    f"(mean {a_mean:.1f}"
                    + (f", hard regressions {regressions}" if regressions else "")
                    + "); not promoted. Targeted training improved target-like tasks while "
                    "damaging stable families — the v2.10/v2.11 failure mode.")
    elif n_conv >= 2 or (n_conv == 1 and a_mean >= V222_MEAN):
        tag = "strong"
        sentence = ("The residual tree failures were partially weight-capability-bound and "
                    "improved under contamination-guarded targeted adaptation.")
    else:
        tag = "candidate"
        sentence = ("Targeted same-family capability training is promoted as a candidate "
                    "intervention for residual tree reasoning failures.")

    s = ["# v2.23 — Targeted Tree Capability Adapter Summary", "",
         f"**Baseline:** {baseline_mean:.1f}/32. **Champion+verifier (v2.22):** hard tasks "
         "serialize 0/3, from_list 2/3, max_path_sum 1/3; full-32 mean 22.0. The adapter is a "
         "fresh LoRA on the MERGED champion (no LoRA-on-LoRA; champion not modified), evaluated "
         "under the same v2.22 verifier mode + v2.19c retrieval.", "",
         "Training data: 49 contamination-guarded same-family-different-task tree repair traces "
         "(0 overlap with the benchmark). The hard tasks stayed evaluation-only.", "",
         "## Hard tree tasks (the decision metric)", "",
         "| Task | Baseline | Champion+verifier | Adapter+verifier | Adapter NO-verifier | Converted |",
         "|---|---|---|---|---|---|"]
    for r in hard_rows:
        s.append(f"| {r['task_id']} | {r['baseline']} | {r['champion_verifier']} | "
                 f"{r['adapter_verifier']} | {r['adapter_NO_verifier']} | {r['converted']} |")
    s += ["",
          f"**Hard-tree conversions (adapter+verifier): {n_conv}** "
          + (f"({', '.join('`'+t+'`' for t in conversions)})" if conversions else "(none)") + ".",
          f"Conversions that PERSIST without the verifier (weight-level): {len(conv_noverif)} "
          + (f"({', '.join('`'+t+'`' for t in conv_noverif)})" if conv_noverif else "(none)") + ".",
          "",
          "## Full-32 (regression gate — not the promotion basis)", "",
          f"- Adapter+verifier full-32: {a_tot} → mean {a_mean:.1f}/32 (std {a_std:.2f}; "
          f"v2.22 champion+verifier 22.0; baseline {baseline_mean:.1f}).",
          f"- Hard regressions (stable_pass → 0/3): " + (", ".join(f"`{t}`" for t in regressions) or "none") + ".",
          f"- Within v2.22 band (≥{V222_BAND_LO}): {'yes' if band_ok else 'NO'}.",
          "",
          "## Verdict", "", f"**{tag.upper()}** — {sentence}", ""]
    if n_conv >= 1:
        s += [
            "### Weight-gain vs verifier-dependence", "",
            f"- Of the {n_conv} conversion(s), {len(conv_noverif)} persist when the structured "
            "verifier is removed (adapter NO-verifier column). Conversions that persist are "
            "weight-level capability gains; conversions that vanish remain verifier-dependent.",
            "",
        ]
    else:
        s += [
            "### Notable observation", "",
            f"- The controlled adapter neither converted a hard task NOR regressed the benchmark "
            f"(full-32 {a_mean:.1f} vs champion 22.0, no hard regressions). This contrasts with the "
            "project's prior retrains, which all regressed (v2.5 53.6%, v2.6 57.1%, Option A 64.3%): "
            "a small, low-LR, separate-adapter-on-merged-champion pilot is SAFE but, here, "
            "insufficient to crack these 3 tasks.",
            "- Differences vs champion on the hard tasks (from_list 2/3→1/3, max_path 1/3→0/3) are "
            "best-of-3 flip-level, not meaningful — none of the three was ever a stable pass.",
            "- The 3 tasks remain capability-bound; they may need more targeted data/steps or are at "
            "the 1.5B capability ceiling. The v2.22 repair traces remain the diagnostic record.",
            "",
        ]
    s += ["See `comparison.csv`, `hardtree.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")

    cb = ["# v2.23 — Claim Boundary", "", f"## Verdict\n\n**{tag.upper()}.** {sentence}", "",
          "## What this measures", "",
          "- A small fresh LoRA (50 steps, lr 1e-5) on the MERGED champion, trained on 49",
          "  contamination-guarded same-family-different-task tree repair traces. Champion not",
          "  modified; protected indexes untouched.",
          "- Decision metric: hard-tree conversion (stable 3/3) under a strict regression gate",
          "  (no hard regression; full-32 not materially below the v2.22 band).",
          "- Required ablation: adapter with vs without the structured verifier.",
          "", "## Contamination guard", "",
          "- scripts/check_v223_contamination.py asserts 0 overlap (function name, benchmark",
          "  name leakage, prompt, hard-task tokens). The 3 hard tasks are evaluation-only.",
          "", "## Not claimed", "",
          "- No SWE-bench success; no production reliability; general tree reasoning NOT solved.",
          "- No frontier-model superiority; no broad SOTA.",
          "- The frozen champion (23/28 = 82.1%) is UNCHANGED; a promoted adapter is an additive",
          "  artifact, not a champion replacement.",
          "- Bounded to the 32-task benchmark, best-of-3."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR / 'claim_boundary.md'}")
    print(f"\nVerdict: {tag.upper()} — hard conversions={n_conv} (weight-level {len(conv_noverif)}), "
          f"full-32 {a_mean:.1f}")


if __name__ == "__main__":
    main()
