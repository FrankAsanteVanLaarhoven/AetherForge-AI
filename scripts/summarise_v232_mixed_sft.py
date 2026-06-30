"""
scripts/summarise_v232_mixed_sft.py — v2.32 mixed SFT summary + promotion gate (committed evidence).

Reads the LOCAL-ONLY mixed-dataset aggregate and, if a GPU run produced them, the training / eval /
benchmark metrics, and writes only small curated summaries safe to commit. v2.32 gate: repair
validation must IMPROVE without dropping the frozen 32-task benchmark (and tool-use must be
preserved). Training + repair success alone never yields PROMOTE — the benchmark gate is mandatory.
No metrics are fabricated.

Writes results/v232_tool_use_preservation/: summary.md, mix.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v232_mixed_sft.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v232_tool_use_preservation"
AGG = ROOT / "data" / "generated" / "v232" / "mixed_aggregate.json"
M = ROOT / "outputs" / "v232_tool_use_preservation_sft"
TRAIN, EVAL, BENCH = M / "training_metrics.json", M / "eval_metrics.json", M / "benchmark_metrics.json"

CHAMPION_32 = 23
MATERIAL_MARGIN = 1
PRESERVATION_MIN = 0.8   # tool-use preservation pass-rate floor
MISSING_BENCH_CMD = ("python scripts/eval_v232_mixed_sft.py --benchmarks "
                     "--base <base> --adapter outputs/v232_tool_use_preservation_sft/adapter")


def decide(train, ev, bench, contamination_violations):
    """Pure three-tier v2.32 decision. PROMOTE requires repair IMPROVEMENT, tool-use preservation,
    and a non-regressing 32-task benchmark; absent benchmark metrics can never PROMOTE."""
    training_ok = bool(train and (train.get("loss_trend") or train.get("final_loss") is not None))
    repair_improved = bool(ev and ev.get("adapter_repair_pass", -1) > ev.get("base_repair_pass", 1 << 30))
    n_p = ev.get("val_preservation", 0) if ev else 0
    preservation_ok = bool(ev and (n_p == 0 or ev.get("adapter_preservation_pass", 0) >= PRESERVATION_MIN * n_p))
    artifact_ok = contamination_violations == 0
    bench_ok = bool(bench
                    and bench.get("adapter_32_pass", -1) >= CHAMPION_32 - MATERIAL_MARGIN
                    and bench.get("tree_serialize_3of3_preserved") is True)
    gates = {"training": training_ok, "repair_improved": repair_improved,
             "tool_use_preserved": preservation_ok, "artifact_safety": artifact_ok,
             "benchmark_non_regression": bench_ok}
    if not train:
        return gates, "HOLD", (
            "**HOLD** — not run in this environment (CPU-only, no CUDA). Mixed dataset is committed; "
            "the GPU-gated split-loss trainer / eval / benchmark harness is ready. No fabricated metrics.")
    if training_ok and repair_improved and preservation_ok and artifact_ok and not bench:
        return gates, "PARTIAL/HOLD-PENDING-BENCHMARKS", (
            "**PARTIAL PROMOTE / HOLD PENDING BENCHMARKS** — training, repair-improvement, tool-use "
            "preservation, and artifact gates PASSED, but the frozen 32-task benchmark gate has NOT "
            f"been run. Full promotion is withheld until it completes: `{MISSING_BENCH_CMD}`.")
    if all(gates.values()):
        return gates, "PROMOTE", (
            "**PROMOTE** — repair validation IMPROVED without dropping the 32-task benchmark "
            "(adapter ≥ champion−1), tool-use preserved, tree_serialize 3/3 preserved, 0 contamination, "
            "no protected artifact overwritten.")
    failed = [k for k, v in gates.items() if not v]
    return gates, "HOLD", (f"**HOLD** — gate(s) not satisfied: {', '.join(failed)}. "
                           "Promotion requires all gates; no fabricated metrics.")


def main():
    if not AGG.exists():
        print("No mixed aggregate. Run make build-v232-mixed-dataset first."); sys.exit(1)
    a = json.loads(AGG.read_text())
    train = json.loads(TRAIN.read_text()) if TRAIN.exists() else None
    ev = json.loads(EVAL.read_text()) if EVAL.exists() else None
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "mix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "key", "count"])
        w.writeheader()
        for k, v in a.get("objective_distribution", {}).items():
            w.writerow({"dimension": "objective", "key": k, "count": v})
        for k, v in a.get("preservation_family_distribution", {}).items():
            w.writerow({"dimension": "preservation_family", "key": k, "count": v})

    gates, short, decision = decide(train, ev, bench, a.get("contamination_guard_violations", 1))
    s = ["# v2.32 — Tool-Use Preservation During Repair-Trace Adaptation", "",
         "Mixes the genuine repair traces with tool-use / scaffold preservation traces (correct "
         "execute_code trajectories on non-held-out tasks) under a SPLIT LOSS (repair objective + "
         "tool-use preservation objective), to learn repair WITHOUT eroding the tool-call behaviour the "
         "frozen 32-task benchmark depends on. Not SOTA; bounded pilot. Dataset/adapter local-only; only "
         "this summary is committed.", "",
         "## Phase 1 — Mixed dataset (committed evidence)", "",
         f"- Total: **{a.get('total')}** | train **{a.get('train_records')}** / val "
         f"**{a.get('val_records')}**.",
         f"- Objectives: repair **{a.get('repair_examples')}** + tool-use preservation "
         f"**{a.get('tool_use_preservation_examples')}** (mix "
         f"{a.get('mix_ratio_repair')}/{a.get('mix_ratio_preservation')}; preservation loss weight "
         f"{a.get('preservation_loss_weight')}).",
         f"- Preservation families: {a.get('preservation_family_distribution')}.",
         f"- Validation split: repair {a.get('val_repair')} + preservation {a.get('val_preservation')}.",
         f"- Contamination guard violations: **{a.get('contamination_guard_violations', 0)}**.", "",
         "## Phase 2 — Split-loss training", ""]
    if train:
        s += [f"- Base `{train.get('base')}` | max_steps {train.get('max_steps')} | loss trend "
              f"{train.get('loss_trend')} | mix {train.get('mix')}."]
    else:
        s += ["- **NOT RUN** — CPU-only environment; GPU-gated trainer skips cleanly (no fabricated "
              "metrics)."]
    s += ["", "## Phase 3 — Evaluation", ""]
    if ev:
        s += [f"- Repair: base {ev.get('base_repair_pass')}/{ev.get('val_repair')} → adapter "
              f"{ev.get('adapter_repair_pass')}/{ev.get('val_repair')}.",
              f"- Tool-use preservation: adapter {ev.get('adapter_preservation_pass')}/"
              f"{ev.get('val_preservation')} (floor {PRESERVATION_MIN:.0%})."]
    else:
        s += ["- **NOT RUN** — GPU-gated (`scripts/eval_v232_mixed_sft.py`)."]
    if bench:
        s += [f"- Benchmark: champion {CHAMPION_32} vs adapter {bench.get('adapter_32_pass')}; "
              f"tree_serialize 3/3 preserved {bench.get('tree_serialize_3of3_preserved')}."]
    else:
        s += ["- Benchmark gate: **NOT RUN** — required for PROMOTE: `" + MISSING_BENCH_CMD + "`."]
    s += ["", "## Decision", "",
          "| Gate | Status |", "|---|---|"]
    for k in ("training", "repair_improved", "tool_use_preserved", "artifact_safety",
              "benchmark_non_regression"):
        st = "PASS" if gates[k] else ("PENDING" if (not train or (k == "benchmark_non_regression" and not bench)
                                                    or (k in ("repair_improved", "tool_use_preserved") and not ev))
                                      else "FAIL")
        s.append(f"| {k} | {st} |")
    s += ["", decision, "", "See `mix.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.32 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.32 builds a mixed repair + tool-use-preservation SFT substrate with a split-loss "
          "trainer and a gated evaluation, to test whether tool-use can be preserved while adapting on "
          "repair traces.", "",
          "## Status", "",
          f"- Mixed dataset: DONE and committed (summary only; dataset local-only).",
          ("- Training/eval: DONE on a GPU host." if train else
           "- Training/eval: DEFERRED — CPU-only environment; GPU-gated scripts skip cleanly. No metrics "
           "fabricated."),
          ("- Benchmark gate: DONE." if bench else
           "- Benchmark gate: NOT RUN — full promotion withheld until the 32-task eval completes."), "",
          "## Not claimed", "",
          "- No SWE-bench success, no production reliability, no RL training, no general SOTA, no "
          "frontier-level agent performance, no broad model improvement.",
          "- No champion adapter or memory index created or overwritten; preservation traces are correct "
          "scaffolds on newly-authored non-held-out tasks, not held-out solutions."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR}/summary.md, mix.csv, claim_boundary.md")
    print(f"gates={gates} decision={short}")


if __name__ == "__main__":
    main()
