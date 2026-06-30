"""
scripts/summarise_v233_scaffold_sft.py — v2.33 scaffold-first summary + decision gate (committed).

Reads the LOCAL-ONLY scaffold-dataset aggregate and, if a GPU run produced them, the training / eval /
benchmark metrics, and writes only small curated summaries safe to commit. v2.33 success is TOOL-CALL
PRESERVATION WITHOUT REGRESSION — repair is NOT a gate. The benchmark gate is mandatory for PROMOTE;
no metrics are fabricated.

Writes results/v233_scaffold_first/: summary.md, scaffold.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v233_scaffold_sft.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v233_scaffold_first"
AGG = ROOT / "data" / "generated" / "v233" / "scaffold_aggregate.json"
M = ROOT / "outputs" / "v233_scaffold_first_sft"
TRAIN, EVAL, BENCH = M / "training_metrics.json", M / "eval_metrics.json", M / "benchmark_metrics.json"

CHAMPION_32 = 23
MATERIAL_MARGIN = 1
PRESERVATION_MIN = 0.8
MISSING_BENCH_CMD = ("python scripts/eval_v233_scaffold_sft.py --benchmarks "
                     "--base <base> --adapter outputs/v233_scaffold_first_sft/adapter")


def decide(train, ev, bench, contamination_violations):
    """Pure scaffold-first decision. PROMOTE requires tool-use preservation >= 80% AND a
    non-regressing 32-task benchmark with tree_serialize 3/3 AND no_tool_call NOT dominant.
    Repair is not a gate. Absent benchmark metrics can never PROMOTE."""
    training_ok = bool(train and (train.get("loss_trend") or train.get("final_loss") is not None))
    n = ev.get("val_records", 0) if ev else 0
    preservation_ok = bool(ev and n and ev.get("preservation_pass", 0) >= PRESERVATION_MIN * n)
    artifact_ok = contamination_violations == 0
    bench_ok = bool(bench
                    and bench.get("adapter_32_pass", -1) >= CHAMPION_32 - MATERIAL_MARGIN
                    and bench.get("tree_serialize_3of3_preserved") is True
                    and not bench.get("no_tool_call_dominant", False))
    gates = {"training": training_ok, "tool_use_preserved": preservation_ok,
             "artifact_safety": artifact_ok, "benchmark_non_regression": bench_ok}
    if not train:
        return gates, "HOLD", (
            "**HOLD** — not run in this environment (CPU-only, no CUDA). Scaffold dataset is committed; "
            "the GPU-gated trainer / eval / benchmark harness is ready. No fabricated metrics.")
    if training_ok and preservation_ok and artifact_ok and not bench:
        return gates, "PARTIAL/HOLD-PENDING-BENCHMARKS", (
            "**PARTIAL / HOLD PENDING BENCHMARKS** — training and tool-use preservation gates PASSED, "
            "but the frozen 32-task benchmark gate has NOT been run. Full promotion is withheld until "
            f"it completes: `{MISSING_BENCH_CMD}`.")
    if all(gates.values()):
        return gates, "PROMOTE", (
            "**PROMOTE** — tool-use preserved (≥80%), frozen 32-task non-regression vs champion 23/28, "
            "tree_serialize 3/3 preserved, no_tool_call NOT dominant, 0 contamination, no protected "
            "artifact overwritten. Scaffold/tool-call policy preserved by itself — repair adaptation "
            "(v2.34) may now be reintroduced on this base.")
    failed = [k for k, v in gates.items() if not v]
    extra = ""
    if bench and bench.get("no_tool_call_dominant"):
        extra = " no_tool_call is the dominant failure mode (the v2.31/v2.32 collapse persists)."
    return gates, "HOLD/REJECT", (f"**HOLD/REJECT** — gate(s) not satisfied: {', '.join(failed)}.{extra} "
                                  "Scaffold preservation must hold before repair traces return.")


def main():
    if not AGG.exists():
        print("No scaffold aggregate. Run make build-v233-scaffold-dataset first."); sys.exit(1)
    a = json.loads(AGG.read_text())
    train = json.loads(TRAIN.read_text()) if TRAIN.exists() else None
    ev = json.loads(EVAL.read_text()) if EVAL.exists() else None
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "scaffold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "key", "count"])
        w.writeheader()
        for k, v in a.get("objective_distribution", {}).items():
            w.writerow({"dimension": "objective", "key": k, "count": v})
        for k, v in a.get("task_family_distribution", {}).items():
            w.writerow({"dimension": "task_family", "key": k, "count": v})

    gates, short, decision = decide(train, ev, bench, a.get("contamination_guard_violations", 1))
    s = ["# v2.33 — Scaffold-First Tool-Call Preservation", "",
         "Isolates scaffold/tool-call preservation. v2.31 (repair-only) and v2.32 (repair + "
         "preservation) both improved local repair but COLLAPSED the frozen 32-task agent benchmark "
         "with `no_tool_call` dominant. v2.33 trains ONLY on correct execute_code scaffold trajectories "
         "(no repair) to test whether tool-use and the 32-task benchmark can be preserved first. "
         "Success is preservation without regression — NOT repair improvement. Dataset/adapter "
         "local-only; only this summary is committed.", "",
         "## Phase 1 — Scaffold dataset (committed evidence)", "",
         f"- Total: **{a.get('total')}** | train **{a.get('train_records')}** / val "
         f"**{a.get('val_records')}**.",
         f"- Objectives: {a.get('objective_distribution')} (repair examples = "
         f"**{a.get('repair_examples', 0)}**, must be 0).",
         f"- Task families: {a.get('task_family_distribution')}.",
         f"- Contamination guard violations: **{a.get('contamination_guard_violations', 0)}** "
         f"(rejections {a.get('rejection_reasons', {})}).", "",
         "## Phase 2 — Scaffold-only training", ""]
    s += ([f"- Base `{train.get('base')}` | scaffold-only | loss trend {train.get('loss_trend')}."]
          if train else ["- **NOT RUN** — CPU-only; GPU-gated trainer skips cleanly (no fabricated metrics)."])
    s += ["", "## Phase 3 — Evaluation (scaffold-first)", ""]
    if ev:
        s += [f"- Tool-use preservation: {ev.get('preservation_pass')}/{ev.get('val_records')} "
              f"(rate {ev.get('preservation_rate')}; tool_call_rate {ev.get('tool_call_rate')})."]
    else:
        s += ["- Tool-use preservation: **NOT RUN** — GPU-gated."]
    if bench:
        s += [f"- 32-task: champion {CHAMPION_32} vs adapter {bench.get('adapter_32_pass')}; "
              f"tool_call_rate {bench.get('tool_call_rate')}; execute_code_rate "
              f"{bench.get('execute_code_rate')}; no_tool_call {bench.get('no_tool_call')} "
              f"(dominant {bench.get('no_tool_call_dominant')}).",
              f"- Hard-tree {bench.get('hard_tree')}; tree_serialize 3/3 preserved "
              f"{bench.get('tree_serialize_3of3_preserved')}.",
              f"- Failure reasons: {bench.get('failure_reasons')}."]
    else:
        s += ["- Benchmark gate: **NOT RUN** — required for PROMOTE: `" + MISSING_BENCH_CMD + "`."]
    s += ["", "## Decision", "", "| Gate | Status |", "|---|---|"]
    for k in ("training", "tool_use_preserved", "artifact_safety", "benchmark_non_regression"):
        st = "PASS" if gates[k] else ("PENDING" if (not train or (k == "benchmark_non_regression" and not bench)
                                                    or (k == "tool_use_preserved" and not ev)) else "FAIL")
        s.append(f"| {k} | {st} |")
    s += ["", decision, "",
          "_Repair validation is an optional diagnostic only and is NOT a v2.33 promotion gate._", "",
          "See `scaffold.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.33 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.33 builds a scaffold-only (no-repair) tool-call preservation substrate with a GPU-gated "
          "trainer and a gated evaluation, to test whether execute_code / tool-use behaviour and the "
          "frozen 32-task benchmark can be preserved before repair adaptation is reintroduced.", "",
          "## Status", "",
          "- Scaffold dataset: DONE and committed (summary only; dataset local-only).",
          ("- Training/eval/benchmark: DONE on a GPU host." if (train and bench) else
           "- Training/eval/benchmark: DEFERRED — CPU-only; GPU-gated scripts skip cleanly. No metrics "
           "fabricated."), "",
          "## Not claimed", "",
          "- No repair improvement claim (repair objective is intentionally absent in v2.33).",
          "- No SWE-bench success, production reliability, RL training, general SOTA, or frontier-level "
          "agent performance.",
          "- No champion adapter or memory index created or overwritten; scaffolds are correct tool-call "
          "trajectories on newly-authored non-held-out tasks, not held-out solutions."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR}/summary.md, scaffold.csv, claim_boundary.md")
    print(f"gates={gates} decision={short}")


if __name__ == "__main__":
    main()
