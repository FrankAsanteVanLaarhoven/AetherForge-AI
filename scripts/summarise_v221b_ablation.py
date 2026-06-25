"""
scripts/summarise_v221b_ablation.py

Summarise v2.21b — tree-width planning ablation.

Re-runs the v2.21 execution-plan setup with the worked example REMOVED, to test whether the
v2.21 tree_width conversion (6/6) came from the plan structure or from the worked example.
Compares tree_width under ablation against the v2.21 full-prompt result, reports the other
tree stable-fails, aggregate, plan-adherence guard metrics, and a decision per the spec.

Writes results/v221b_tree_width_ablation/: summary.md, comparison.csv, tree_stablefails.csv,
guard_metrics.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v221b_ablation.py
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.summarise_v219_structured_memory import load_baseline  # noqa: E402
from scripts.summarise_v221_reasoning import load_rows, passed, pass_map, guard_flags  # noqa: E402

OUT_DIR = Path("results/v221b_tree_width_ablation")
TREE_STABLEFAILS = ["v210_tree_from_list", "v210_tree_max_path_sum",
                    "v210_tree_serialize", "v210_tree_width"]
ABL_SUBSET = [Path(f"outputs/eval_v221b_tree_stablefails_run{n}") for n in range(1, 7)]
ABL_FULL = [Path(f"outputs/eval_v221b_reasoning_tree_32_run{n}") for n in (1, 2, 3)]
# v2.21 full-prompt references (subset 3 + full 3 = up to 6 runs for tree_width)
V221_SUBSET = [Path(f"outputs/eval_v221_tree_stablefails_run{n}") for n in (1, 2, 3)]
V221_FULL = [Path(f"outputs/eval_v221_reasoning_tree_32_run{n}") for n in (1, 2, 3)]


def task_passes(dirs, task):
    n = c = 0
    for d in dirs:
        rows = load_rows(d)
        if not rows:
            continue
        for row in rows:
            if (row.get("id") or row.get("task_id")) == task:
                n += 1
                c += int(passed(row))
    return c, n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_npass, base_stab, base_totals = load_baseline()
    baseline_mean = sum(base_totals) / len(base_totals)

    if not any((d / "best_of_3.csv").exists() for d in ABL_SUBSET + ABL_FULL):
        print("No v2.21b ablation data found. Run the eval-v221b-* targets first.")
        sys.exit(1)

    # tree_width under ablation: prefer the 6 subset runs; also report full-32.
    tw_sub_c, tw_sub_n = task_passes(ABL_SUBSET, "v210_tree_width")
    tw_full_c, tw_full_n = task_passes(ABL_FULL, "v210_tree_width")
    # v2.21 reference (subset + full)
    v221_tw_c, v221_tw_n = task_passes(V221_SUBSET + V221_FULL, "v210_tree_width")

    # ── tree table (ablation subset + full, vs v2.21) ────────────────────────
    rows = []
    for t in TREE_STABLEFAILS:
        a_sc, a_sn = task_passes(ABL_SUBSET, t)
        a_fc, a_fn = task_passes(ABL_FULL, t)
        v_c, v_n = task_passes(V221_SUBSET + V221_FULL, t)
        rows.append({"task_id": t, "baseline": f"{base_npass.get(t)}/3",
                     "v221_full_prompt": f"{v_c}/{v_n}",
                     "v221b_ablation_subset": f"{a_sc}/{a_sn}",
                     "v221b_ablation_full32": f"{a_fc}/{a_fn}"})
    with open(OUT_DIR / "tree_stablefails.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "baseline", "v221_full_prompt",
                                          "v221b_ablation_subset", "v221b_ablation_full32"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {OUT_DIR / 'tree_stablefails.csv'}")

    # ── aggregate + regressions (full-32 under ablation) ─────────────────────
    full_pt, full_totals = pass_map(ABL_FULL)
    full_mean = statistics.mean(full_totals) if full_totals else 0.0
    regressions = [t for t in base_npass
                   if base_stab.get(t) == "stable_pass" and full_pt.get(t, 0) == 0] if full_totals else []

    # ── guard metrics ────────────────────────────────────────────────────────
    agg = {"plan": 0, "base_case": 0, "combine": 0, "minimal_test": 0, "repair": 0}
    n_traj = 0
    for d in ABL_SUBSET + ABL_FULL:
        for row in load_rows(d):
            fl = guard_flags(row.get("full_transcript") or row.get("assistant_text") or "")
            for k in agg:
                agg[k] += int(fl[k])
            n_traj += 1
    with open(OUT_DIR / "guard_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "n_trajectories", "rate"])
        for k in ["plan", "base_case", "combine", "minimal_test", "repair"]:
            w.writerow([k, agg[k], n_traj, f"{agg[k]/n_traj:.2f}" if n_traj else "0"])
    print(f"Wrote {OUT_DIR / 'guard_metrics.csv'}")

    # combined tree_width rate across all ablation runs (subset + full)
    tw_c, tw_n = tw_sub_c + tw_full_c, tw_sub_n + tw_full_n
    tw_rate = tw_c / tw_n if tw_n else 0.0
    if tw_rate >= 0.83:
        tag = "survives"
        sentence = ("The tree_width conversion SURVIVES removal of the worked example, "
                    "strengthening the claim that structured execution planning improved a "
                    "reasoning-control-bound tree task.")
    elif tw_rate >= 0.33:
        tag = "partial"
        sentence = ("The tree_width conversion is PARTLY example-assisted; execution planning "
                    "remains useful but is not sufficient on its own.")
    else:
        tag = "collapses"
        sentence = ("The tree_width conversion DEPENDS on the worked example; planning alone is "
                    "not yet enough evidence for broad tree reasoning improvement.")

    with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "tree_width_pass", "n_runs", "rate", "aggregate_mean"])
        w.writerow(["v221_full_prompt", v221_tw_c, v221_tw_n,
                    f"{v221_tw_c/v221_tw_n:.2f}" if v221_tw_n else "n/a", "18.3"])
        w.writerow(["v221b_ablation", tw_c, tw_n, f"{tw_rate:.2f}",
                    f"{full_mean:.2f}" if full_totals else "n/a"])
    print(f"Wrote {OUT_DIR / 'comparison.csv'}")

    pr = agg["plan"] / n_traj if n_traj else 0.0
    s = ["# v2.21b — Tree-Width Planning Ablation Summary", "",
         "Execution-plan prompt with the worked example REMOVED (same contract, same v2.19c",
         "retrieval). Tests whether the v2.21 tree_width conversion came from the plan structure",
         "or the example.", "",
         "## tree_width (the ablation target)", "",
         f"- v2.21 full prompt: **{v221_tw_c}/{v221_tw_n}**",
         f"- v2.21b ablation (no example): **{tw_c}/{tw_n}** (rate {tw_rate:.0%}) "
         f"[subset {tw_sub_c}/{tw_sub_n}, full-32 {tw_full_c}/{tw_full_n}]", "",
         "## All tree stable-fails", "",
         "| Task | Baseline | v2.21 full | v2.21b subset | v2.21b full-32 |",
         "|---|---|---|---|---|"]
    for r in rows:
        s.append(f"| {r['task_id']} | {r['baseline']} | {r['v221_full_prompt']} | "
                 f"{r['v221b_ablation_subset']} | {r['v221b_ablation_full32']} |")
    s += ["",
          f"Aggregate full-32 under ablation: {full_totals} → mean {full_mean:.1f}/32 "
          f"(baseline {baseline_mean:.1f}, v2.21 18.3).",
          f"Hard regressions vs baseline: " + (", ".join(f"`{t}`" for t in regressions) or "none") + ".",
          "",
          "## Plan-adherence guard metrics", "",
          f"Over {n_traj} trajectories: PLAN {pr:.0%}, base-case {agg['base_case']/max(n_traj,1):.0%}, "
          f"combine {agg['combine']/max(n_traj,1):.0%}, minimal-test {agg['minimal_test']/max(n_traj,1):.0%}, "
          f"repair {agg['repair']/max(n_traj,1):.0%}.",
          "",
          "## Verdict", "", f"**{tag.upper()}** — {sentence}", "",
          "See `comparison.csv`, `tree_stablefails.csv`, `guard_metrics.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")

    cb = ["# v2.21b — Claim Boundary", "", f"## Verdict\n\n**{tag.upper()}.** {sentence}", "",
          "## What this measures", "",
          "- The v2.21 execution-plan prompt with ONLY the worked example removed (contract,",
          "  retrieval, model, and all else identical) — a clean single-variable ablation.",
          "- Decision metric: tree_width pass rate under ablation vs the v2.21 6/6 result.",
          "", "## Not claimed", "",
          "- No SWE-bench success; no production reliability; no model-weight change.",
          "- No frontier-model superiority; general tree reasoning is NOT solved.",
          "- The other tree stable-fails remain capability-bound (fail despite plan adherence).",
          "- Bounded to the 32-task benchmark, best-of-3.",
          "- No AI/tool/vendor attribution."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR / 'claim_boundary.md'}")
    print(f"\nVerdict: {tag.upper()} — tree_width {tw_c}/{tw_n} ({tw_rate:.0%}) under ablation, PLAN {pr:.0%}")


if __name__ == "__main__":
    main()
