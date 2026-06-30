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


def main():
    if not AGG.exists():
        print("No SFT aggregate. Run make build-v231-sft-dataset first.")
        sys.exit(1)
    a = json.loads(AGG.read_text())
    train = json.loads(TRAIN.read_text()) if TRAIN.exists() else None
    ev = json.loads(EVAL.read_text()) if EVAL.exists() else None
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
    s += ["", "## Phase 3 — Evaluation", ""]
    if ev:
        s += [f"- Repair validation: base {ev.get('base_repair_pass')}/{ev.get('val_records')} vs "
              f"adapter {ev.get('adapter_repair_pass')}/{ev.get('val_records')} "
              f"(base {ev.get('base_rate')} → adapter {ev.get('adapter_rate')}).",
              "- 32-task / hard-tree / tree_serialize-3/3 delegated to evaluate_code_agent.py "
              "(compare vs frozen champion 23/28)."]
    else:
        s += ["- **NOT RUN** — GPU-gated (`scripts/eval_v231_repair_sft.py`). Planned: (1) v2.30 repair "
              "validation slice; (2) frozen 32-task benchmark; (3) hard tree subset; (4) tree_serialize "
              "3/3 format-control check. Comparisons: base+verifier vs adapter+verifier vs adapter "
              "without verifier. No evaluation metrics are fabricated."]
    promote = bool(train and ev and a.get("contamination_guard_violations", 1) == 0)
    s += ["", "## Decision", "",
          ("**PROMOTE** — see training/eval metrics above; gates met." if promote else
           "**HOLD** — dataset export + GPU-gated trainer/eval harness are delivered and validated "
           "offline (export runs; trainer/eval skip cleanly with no fabricated metrics), but the "
           "actual pilot run is deferred to a GPU host. Promotion requires a stable training run + "
           "non-regressing 32-task benchmark, which this CPU-only environment cannot produce."), "",
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
          "- Training + evaluation: DEFERRED — this environment is CPU-only (no CUDA); the trainer and "
          "eval scripts skip cleanly and fabricate no metrics. Run on a GPU host to complete the pilot.",
          "", "## Not claimed", "",
          "- No SWE-bench success, no production reliability, no RL training, no general SOTA, no "
          "frontier-level agent performance, no broad model improvement.",
          "- No model weights, champion adapters, or memory indexes were created or overwritten.",
          "- The traces are controlled perturbations of non-held-out functions; the pilot tests repair "
          "supervision, not new capability."]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")

    print(f"Wrote {OUT_DIR}/summary.md, dataset.csv, claim_boundary.md")
    print(f"train={a.get('train_records')} val={a.get('val_records')} "
          f"trained={'yes' if train else 'no(GPU-gated)'} evaluated={'yes' if ev else 'no(GPU-gated)'} "
          f"decision={'PROMOTE' if promote else 'HOLD'}")


if __name__ == "__main__":
    main()
