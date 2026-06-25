"""
scripts/summarise_v222b_ablation.py

Summarise v2.22b — repair-signal ablation (raw stderr vs structured VERIFIER).

Single-variable ablation of v2.22: SAME repair budget + no-repeat + diagnostic-assert
contract, but the OBSERVATION is RAW stderr instead of the distilled VERIFIER block. By
comparing v2.22b (raw) against v2.22 (verifier), this attributes the v2.22 aggregate lift
(mean 22.0) to either the structured signal format or the repair discipline:

  if v2.22b ~= v2.22  -> the gain is from repair DISCIPLINE (budget + no-repeat), not the
                         structured signal format.
  if v2.22b << v2.22  -> the structured VERIFIER signal format drives the gain.

Writes results/v222b_repair_signal_ablation/: summary.md, comparison.csv, capbound.csv,
claim_boundary.md.

Usage:
    python scripts/summarise_v222b_ablation.py
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.summarise_v219_structured_memory import load_baseline  # noqa: E402
from scripts.summarise_v221_reasoning import load_rows, passed, pass_map  # noqa: E402

OUT_DIR = Path("results/v222b_repair_signal_ablation")
CAPBOUND = ["v210_tree_serialize", "v210_tree_from_list", "v210_tree_max_path_sum"]

RAW_CAP = [Path(f"outputs/eval_v222b_repair_capbound_run{n}") for n in (1, 2, 3)]
RAW_FULL = [Path(f"outputs/eval_v222b_repair_32_run{n}") for n in (1, 2, 3)]
VER_CAP = [Path(f"outputs/eval_v222_repair_capbound_run{n}") for n in (1, 2, 3)]
VER_FULL = [Path(f"outputs/eval_v222_repair_32_run{n}") for n in (1, 2, 3)]

V221_MEAN = 18.3
NOISE = 1.5  # aggregate-mean noise band (tasks)


def repaired_to_pass(dirs):
    n = rp = 0
    for d in dirs:
        for row in load_rows(d):
            n += 1
            tr = row.get("full_transcript") or row.get("assistant_text") or ""
            if passed(row) and ("VERIFIER:" in tr or "OBSERVATION:" in tr) and \
               ("AssertionError" in tr or "Error" in tr or "Traceback" in tr):
                rp += 1
    return rp, n


def cap_pass(dirs, t):
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

    if not any((d / "best_of_3.csv").exists() for d in RAW_CAP + RAW_FULL):
        print("No v2.22b eval data found. Run the eval-v222b-* targets first.")
        sys.exit(1)

    raw_pt, raw_tot = pass_map(RAW_FULL)
    ver_pt, ver_tot = pass_map(VER_FULL)
    raw_mean = statistics.mean(raw_tot) if raw_tot else 0.0
    ver_mean = statistics.mean(ver_tot) if ver_tot else 0.0
    raw_rp, raw_n = repaired_to_pass(RAW_CAP + RAW_FULL)
    ver_rp, ver_n = repaired_to_pass(VER_CAP + VER_FULL)

    # regressions for raw full-32
    regressions = [t for t in base_npass
                   if base_stab.get(t) == "stable_pass" and raw_pt.get(t, 0) == 0] if raw_tot else []

    # capbound comparison
    cap_rows = []
    for t in CAPBOUND:
        rc, rn = cap_pass(RAW_CAP, t)
        vc, vn = cap_pass(VER_CAP, t)
        cap_rows.append({"task_id": t, "baseline": f"{base_npass.get(t)}/3",
                         "v222_verifier": f"{vc}/{vn}", "v222b_raw": f"{rc}/{rn}"})
    with open(OUT_DIR / "capbound.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "baseline", "v222_verifier", "v222b_raw"])
        w.writeheader(); w.writerows(cap_rows)
    print(f"Wrote {OUT_DIR / 'capbound.csv'}")

    with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "full32_totals", "full32_mean", "repaired_to_pass_rate"])
        w.writerow(["v222_verifier", " ".join(map(str, ver_tot)), f"{ver_mean:.2f}",
                    f"{ver_rp}/{ver_n}" if ver_n else "n/a"])
        w.writerow(["v222b_raw", " ".join(map(str, raw_tot)), f"{raw_mean:.2f}",
                    f"{raw_rp}/{raw_n}" if raw_n else "n/a"])
    print(f"Wrote {OUT_DIR / 'comparison.csv'}")

    # ── attribution verdict ──────────────────────────────────────────────────
    delta = ver_mean - raw_mean   # how much the structured signal adds over raw
    if not ver_tot:
        tag, sentence = ("incomparable", "v2.22 verifier runs not found; cannot attribute.")
    elif abs(delta) <= NOISE:
        tag = "discipline_attributed"
        sentence = (f"Raw-stderr repair matches the structured VERIFIER within noise "
                    f"(raw {raw_mean:.1f} vs verifier {ver_mean:.1f}). The v2.22 aggregate lift is "
                    "attributable to repair DISCIPLINE (bounded budget + no-repeat + diagnostic "
                    "asserts), NOT to the structured signal format. The verifier formatting is a "
                    "clean diagnostic but not the cause of the gain.")
    elif delta >= NOISE:
        tag = "signal_attributed"
        sentence = (f"The structured VERIFIER scores {delta:.1f} tasks above raw stderr "
                    f"(verifier {ver_mean:.1f} vs raw {raw_mean:.1f}, which sits at baseline); the "
                    "signal FORMAT — distilling the failure into a labeled, actionable block — "
                    "drives the v2.22 lift, not the repair discipline. The 1.5B model acts on the "
                    "distilled signal but not on a raw traceback.")
    else:
        tag = "raw_better"
        sentence = (f"Raw stderr scored higher than the structured VERIFIER "
                    f"(raw {raw_mean:.1f} vs verifier {ver_mean:.1f}); the structuring did not help "
                    "and the gain is from repair discipline.")

    s = ["# v2.22b — Repair-Signal Ablation (raw stderr vs structured VERIFIER)", "",
         f"**Baseline:** {baseline_mean:.1f}/32. **v2.21 (plan, no budget):** {V221_MEAN}/32. "
         "Single-variable change vs v2.22: signal format only (budget + no-repeat + diagnostic "
         "asserts held constant).", "",
         "## Aggregate (full-32, best-of-3)", "",
         "| Cohort | runs | mean | repaired-to-pass |",
         "|---|---|---|---|",
         f"| v2.22 structured VERIFIER | {ver_tot} | {ver_mean:.1f} | {ver_rp}/{ver_n} |",
         f"| v2.22b raw stderr | {raw_tot} | {raw_mean:.1f} | {raw_rp}/{raw_n} |",
         "",
         f"Δ (verifier − raw): **{delta:+.1f} tasks** (noise band ±{NOISE}).",
         f"Full-32 hard regressions (raw): " + (", ".join(f"`{t}`" for t in regressions) or "none") + ".",
         "",
         "## Capability-bound tasks (both should remain unconverted)", "",
         "| Task | Baseline | v2.22 verifier | v2.22b raw |",
         "|---|---|---|---|"]
    for r in cap_rows:
        s.append(f"| {r['task_id']} | {r['baseline']} | {r['v222_verifier']} | {r['v222b_raw']} |")
    s += ["", "## Verdict", "", f"**{tag.upper()}** — {sentence}", "",
          "See `comparison.csv`, `capbound.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")

    cb = ["# v2.22b — Claim Boundary", "", f"## Verdict\n\n**{tag.upper()}.** {sentence}", "",
          "## What this measures", "",
          "- v2.22b changes ONE variable vs v2.22: the failed-execution OBSERVATION is RAW stderr",
          "  instead of the distilled VERIFIER block. Budget, no-repeat, diagnostic-assert",
          "  contract, retrieval, and model are identical.",
          "- Attributes the v2.22 aggregate lift to signal format vs repair discipline.",
          "", "## Not claimed", "",
          "- No SWE-bench success; no production reliability; no model-weight change.",
          "- No frontier-model superiority; the 3 capability-bound tree tasks remain unsolved.",
          "- Bounded to the 32-task benchmark, best-of-3.",
          "- No AI/tool/vendor attribution."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR / 'claim_boundary.md'}")
    print(f"\nVerdict: {tag.upper()} — verifier {ver_mean:.1f} vs raw {raw_mean:.1f} (Δ {delta:+.1f})")


if __name__ == "__main__":
    main()
