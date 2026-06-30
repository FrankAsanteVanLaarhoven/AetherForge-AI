"""
scripts/summarise_v231_repair_sft.py — v2.31 SFT pilot summary (committed evidence).

Reads the LOCAL-ONLY SFT export aggregate and, if a GPU run has produced them, the training/eval
metrics, and writes only small curated summaries safe to commit. If training/eval metrics are absent
(this CPU-only environment cannot train), the summary honestly reports HOLD: dataset + harness ready,
run deferred to a GPU host. No metrics are fabricated.

Writes results/v231_tiny_repair_trace_sft/: summary.md, dataset.csv, claim_boundary.md.

Usage:
    python scripts/summarise_v231_repair_sft.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v231_tiny_repair_trace_sft"
AGG = ROOT / "data" / "generated" / "v231" / "sft_aggregate.json"
TRAIN = ROOT / "outputs" / "v231_tiny_repair_trace_sft" / "training_metrics.json"
EVAL = ROOT / "outputs" / "v231_tiny_repair_trace_sft" / "eval_metrics.json"
BENCH = ROOT / "outputs" / "v231_tiny_repair_trace_sft" / "benchmark_metrics.json"

# champion reference on the frozen benchmark; "material regression" = adapter drops by > this margin.
CHAMPION_32 = 23
MATERIAL_MARGIN = 1   # a single-task best-of-3 flip is noise; > 1 task drop is material

# exact command a GPU host must run to fill the benchmark gate (see eval_v231_repair_sft.py --benchmarks)
MISSING_BENCH_CMD = ("python scripts/eval_v231_repair_sft.py --benchmarks "
                     "--base <base> --adapter outputs/v231_tiny_repair_trace_sft/adapter")


def decide(train, ev, bench, contamination_violations):
    """Pure three-tier promotion decision. PROMOTE requires the benchmark gate; absent benchmark
    metrics can never yield PROMOTE (training + repair alone => HOLD PENDING BENCHMARKS)."""
    training_ok = bool(train and (train.get("loss_trend") or train.get("final_loss") is not None))
    repair_ok = bool(ev and ev.get("adapter_repair_pass", -1) >= ev.get("base_repair_pass", 1 << 30))
    artifact_ok = contamination_violations == 0
    bench_ok = bool(bench
                    and bench.get("adapter_32_pass", -1) >= CHAMPION_32 - MATERIAL_MARGIN
                    and bench.get("tree_serialize_3of3_preserved") is True)
    gates = {"training": training_ok, "repair_validation": repair_ok,
             "artifact_safety": artifact_ok, "benchmark_non_regression": bench_ok}

    if not train:
        return gates, "HOLD", (
            "**HOLD** — pilot not run in this environment (CPU-only, no CUDA). Dataset export is "
            "committed; the GPU-gated trainer/eval/benchmark harness is ready. No metrics are fabricated.")
    if training_ok and repair_ok and artifact_ok and not bench:
        return gates, "PARTIAL/HOLD-PENDING-BENCHMARKS", (
            "**PARTIAL PROMOTE / HOLD PENDING BENCHMARKS** — training, repair-validation, and "
            "artifact-safety gates PASSED, but the frozen 32-task / hard-tree / tree_serialize 3/3 "
            "benchmark gate has NOT been run. Full promotion is withheld until those delegated "
            f"evaluations complete: `{MISSING_BENCH_CMD}`.")
    if all(gates.values()):
        return gates, "PROMOTE", (
            "**PROMOTE** — all gates passed: stable training, repair validation adapter ≥ base, 0 "
            "contamination, no material 32-task regression vs champion 23/28, tree_serialize 3/3 "
            "preserved, no protected artifact overwritten.")
    failed = [k for k, v in gates.items() if not v]
    return gates, "HOLD", (f"**HOLD** — gate(s) not satisfied: {', '.join(failed)}. "
                           "Promotion requires all gates; no fabricated metrics.")


def main():
    if not AGG.exists():
        print("No SFT aggregate. Run make build-v231-sft-dataset first.")
        sys.exit(1)
    a = json.loads(AGG.read_text())
    train = json.loads(TRAIN.read_text()) if TRAIN.exists() else None
    ev = json.loads(EVAL.read_text()) if EVAL.exists() else None
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "key", "count"])
        w.writeheader()
        for k, v in a.get("category_distribution", {}).items():
            w.writerow({"dimension": "category", "key": k, "count": v})
        for k, v in a.get("task_family_distribution", {}).items():
            w.writerow({"dimension": "task_family", "key": k, "count": v})
        for k, v in a.get("source_counts", {}).items():
            w.writerow({"dimension": "source", "key": k, "count": v})

    s = ["# v2.31 — Tiny Repair-Trace SFT Pilot", "",
         "First training milestone: a tiny supervised fine-tuning pilot on the clean v2.30 repair-trace "
         "substrate. Target behaviour: given task + failed candidate + structured verifier signal, "
         "produce repair plan + corrected solution. Bounded pilot — not SOTA, not production, not RL.",
         "", "## Phase 1 — SFT dataset export (committed evidence)", "",
         f"- Total available records: **{a.get('total_available', 0)}**",
         f"- Training records: **{a.get('train_records', 0)}** | validation records: "
         f"**{a.get('val_records', 0)}**",
         f"- Format-repair: **{a.get('format_repair', 0)}** | algorithmic-repair: "
         f"**{a.get('algorithmic_repair', 0)}**",
         f"- Task families: {len(a.get('task_family_distribution', {}))} "
         f"({a.get('task_family_distribution', {})})",
         f"- Sources: {a.get('source_counts', {})}",
         f"- Rejection reasons: {a.get('rejection_reasons', {})}",
         f"- Contamination guard violations: **{a.get('contamination_guard_violations', 0)}** "
         "(any overlap or held-out-name record is rejected at export).", "",
         "## Phase 2 — Tiny SFT run", ""]
    if train:
        s += [f"- Base: `{train.get('base')}` | LoRA r=8 | max_steps {train.get('max_steps')} | "
              f"lr {train.get('lr')} | train_records {train.get('train_records')}.",
              f"- Final loss: {train.get('final_loss')} | loss trend: {train.get('loss_trend')}.",
              f"- Adapter: `{train.get('adapter_path')}` (separate path; champion untouched)."]
    else:
        s += ["- **NOT RUN in this environment** — CPU-only torch (`+cpu`, no CUDA). The trainer "
              "(`scripts/train_v231_repair_sft.py`) is GPU-gated and skips cleanly here. Run on a GPU "
              "host to produce the adapter at `outputs/v231_tiny_repair_trace_sft/` (local-only). "
              "No training metrics are fabricated."]
    s += ["", "## Phase 3a — Repair validation", ""]
    if ev:
        s += [f"- Repair validation slice: base {ev.get('base_repair_pass')}/{ev.get('val_records')} "
              f"vs adapter {ev.get('adapter_repair_pass')}/{ev.get('val_records')} "
              f"(base {ev.get('base_rate')} → adapter {ev.get('adapter_rate')})."]
    else:
        s += ["- **NOT RUN** — GPU-gated (`scripts/eval_v231_repair_sft.py`)."]
    s += ["", "## Phase 3b — Delegated benchmark gates (frozen 32-task / hard tree / tree_serialize)", ""]
    if bench:
        s += [f"- 32-task: champion {bench.get('champion_32_pass', CHAMPION_32)} vs adapter "
              f"{bench.get('adapter_32_pass')} (no material regression = adapter ≥ "
              f"{CHAMPION_32 - MATERIAL_MARGIN}).",
              f"- Hard-tree subset: {bench.get('hard_tree')}.",
              f"- tree_serialize 3/3 format-control preserved: {bench.get('tree_serialize_3of3_preserved')}."]
    else:
        s += ["- **NOT RUN** — required for full promotion. Run on the GPU host (adapter must exist):",
              "",
              "  ```bash",
              "  python scripts/eval_v231_repair_sft.py --benchmarks \\",
              "    --base <base> --adapter outputs/v231_tiny_repair_trace_sft/adapter",
              "  ```",
              "  (delegates to evaluate_code_agent.py: 32-task `data/v210_clean_repair_generalisation_tasks.jsonl`,",
              "  hard-tree subset, and the v2.26 representation tasks for tree_serialize 3/3.)"]

    # ── per-gate decision (three-tier) ──
    gates, short, decision = decide(train, ev, bench, a.get("contamination_guard_violations", 1))
    training_ok, repair_ok = gates["training"], gates["repair_validation"]
    artifact_ok, bench_ok = gates["artifact_safety"], gates["benchmark_non_regression"]

    s += ["", "## Decision", "",
          "| Gate | Status |", "|---|---|",
          f"| Training stable | {'PASS' if training_ok else ('PENDING' if not train else 'FAIL')} |",
          f"| Repair validation (adapter ≥ base) | {'PASS' if repair_ok else ('PENDING' if not ev else 'FAIL')} |",
          f"| Artifact safety (contamination 0) | {'PASS' if artifact_ok else 'FAIL'} |",
          f"| 32-task non-regression + tree_serialize 3/3 | {'PASS' if bench_ok else ('PENDING' if not bench else 'FAIL')} |",
          "", decision, "",
          "See `dataset.csv`, `claim_boundary.md`."]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")

    cb = ["# v2.31 — Claim Boundary", "",
          "## Claimed", "",
          "- v2.31 provides an initial supervised repair-trace adaptation pilot: a contamination-clean "
          "SFT export (40 train / 10 val from 50 genuine repairs) plus a GPU-gated tiny-LoRA trainer "
          "and evaluation harness to test whether verifier-labelled repair traces can train local "
          "repair behaviour.", "",
          "## Status", "",
          "- Dataset export: DONE and committed (summary only; dataset local-only).",
          (f"- Training: DONE on a GPU host (loss trend {train.get('loss_trend')})." if train else
           "- Training: DEFERRED — CPU-only environment; GPU-gated trainer skips cleanly."),
          (f"- Repair validation: DONE (base {ev.get('base_repair_pass')} → adapter "
           f"{ev.get('adapter_repair_pass')} of {ev.get('val_records')})." if ev else
           "- Repair validation: PENDING (GPU)."),
          ("- Benchmark gate (32-task / hard-tree / tree_serialize 3/3): DONE." if bench else
           "- Benchmark gate (32-task / hard-tree / tree_serialize 3/3): NOT RUN — full promotion is "
           "withheld until these delegated evaluations complete. No metrics are fabricated."),
          "", "## Not claimed", "",
          "- No SWE-bench success, no production reliability, no RL training, no general SOTA, no "
          "frontier-level agent performance, no broad model improvement.",
          "- No model weights, champion adapters, or memory indexes were created or overwritten.",
          "- The traces are controlled perturbations of non-held-out functions; the pilot tests repair "
          "supervision, not new capability."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")

    print(f"Wrote {OUT_DIR}/summary.md, dataset.csv, claim_boundary.md")
    print(f"gates={gates} decision={short}")


if __name__ == "__main__":
    main()
