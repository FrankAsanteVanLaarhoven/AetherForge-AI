#!/usr/bin/env python3
"""
scripts/summarise_v29_memory_repair.py
AetherForge v2.9 — Memory Repair Split

Reads diagnostic_repair_results.csv (same benchmark, repair index)
and clean_generalisation_results.csv (untouched test set, repair index),
then produces:
  results/v29_memory_repair/summary.md
  results/v29_memory_repair/claim_boundary.md

Promotion rules:
  diagnostic_repair   — evaluated on the same frozen 28-task benchmark.
                        CANNOT promote a new champion even if score >= 24/28.
                        Only establishes that repair examples help on known tasks.
  clean_generalisation— evaluated on the separate v29_clean_generalisation_tasks.
                        A clean pass/fail result on never-seen tasks.
                        Can support a generalisation claim, not a 28-task promotion.
  New 28-task champion requires:
    (a) score >= 24/28 AND
    (b) evaluated on the original frozen benchmark WITHOUT the repair index
        (i.e. adding repair records to index_adapted and re-running).
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CHAMPION_SCORE       = 23
TOTAL_TASKS          = 28
PROMOTION_THRESHOLD  = 24

REPAIR_TARGETS = ["merge_intervals", "median_two_sorted", "deep_get", "tree_depth_tuple"]

TREE_DEPTH_NOTE = (
    "tree_depth_tuple has a broken assertion in the original task description: "
    "the prompt claims tree_depth(((1,2),(3,(4,5))))==3 but the correct value "
    "by the stated rule (leaves=1, branch=1+max) is 4. "
    "Any correct implementation will fail this assertion if the model copies it verbatim. "
    "The repair record uses the correct value. Diagnostic results for this task "
    "may be affected by the model's choice of whether to follow the task prompt or the repair memory."
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _pass_count(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("passed", "").strip().lower() in ("true", "1", "yes", "pass"))


def _task_pass_map(rows: list[dict]) -> dict[str, bool]:
    return {
        r["task_id"]: r.get("passed", "").strip().lower() in ("true", "1", "yes", "pass")
        for r in rows
        if "task_id" in r
    }


def build_summary(
    diagnostic_rows: list[dict],
    clean_rows: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    diag_n   = len(diagnostic_rows)
    diag_pass = _pass_count(diagnostic_rows)
    diag_pct  = 100.0 * diag_pass / diag_n if diag_n else 0.0
    diag_map  = _task_pass_map(diagnostic_rows)

    clean_n    = len(clean_rows)
    clean_pass = _pass_count(clean_rows)
    clean_pct  = 100.0 * clean_pass / clean_n if clean_n else 0.0

    # ── summary.md ──────────────────────────────────────────────────────────
    lines = [
        "# AetherForge v2.9 — Memory Repair Split: Summary",
        f"\nGenerated: {now}\n",
        "## Context",
        f"- Champion: {CHAMPION_SCORE}/{TOTAL_TASKS} = {100*CHAMPION_SCORE/TOTAL_TASKS:.1f}%",
        f"- Promotion threshold: {PROMOTION_THRESHOLD}/{TOTAL_TASKS}",
        "- Champion index: `memory/index_adapted` (99 records, PROTECTED)",
        "- Repair index:   `memory/index_v29_repair` (champion + 4 repair examples)",
        "- Repair records are NOT merged into the champion index\n",
        "## Diagnostic repair (same 28-task benchmark)",
        "",
    ]

    if not diagnostic_rows:
        lines.append("_No results yet — run `make eval-v29-repair-memory-diagnostic`_\n")
    else:
        lines += [
            f"Score: **{diag_pass}/{diag_n} = {diag_pct:.1f}%** (diagnostic label only)",
            "",
            "⚠️  This run uses the same frozen 28-task benchmark AND adds task-specific",
            "repair records for the 4 failing tasks. It is NOT a clean result.",
            "It demonstrates whether the repair examples reduce known failures.",
            "",
            "### Repair target outcomes",
            "",
            "| Task | Champion result | Diagnostic result | Change |",
            "|------|-----------------|-------------------|--------|",
        ]
        champion_results = {
            "merge_intervals": "FAIL",
            "median_two_sorted": "FAIL",
            "deep_get": "FAIL",
            "tree_depth_tuple": "FAIL",
        }
        for task in REPAIR_TARGETS:
            champ = champion_results.get(task, "FAIL")
            diag  = "PASS" if diag_map.get(task, False) else "FAIL"
            change = "✓ fixed" if champ == "FAIL" and diag == "PASS" else \
                     "✗ still failing" if champ == "FAIL" and diag == "FAIL" else \
                     "~ no change"
            lines.append(f"| {task} | {champ} | {diag} | {change} |")
        lines += [
            "",
            f"**Note on tree_depth_tuple:** {TREE_DEPTH_NOTE}",
            "",
            "### Claim",
            "",
        ]
        if diag_pass >= PROMOTION_THRESHOLD:
            lines += [
                f"Diagnostic score {diag_pass}/{diag_n} meets the promotion threshold.",
                "**However: this is NOT a clean 28-task promotion** because repair records",
                "for the exact failing tasks were added to the index before evaluation.",
                "To promote a new 28-task champion, merge repair records into",
                "`memory/index_adapted` and evaluate on a fresh run without other changes.",
            ]
        else:
            lines += [
                f"Diagnostic score {diag_pass}/{diag_n} = {diag_pct:.1f}% — "
                "repair examples did not raise the score to the promotion threshold.",
            ]

    lines += ["", "## Clean generalisation (separate untouched test set)", ""]

    if not clean_rows:
        lines.append("_No results yet — run `make eval-v29-clean-memory-generalisation`_\n")
    else:
        lines += [
            f"Score: **{clean_pass}/{clean_n} = {clean_pct:.1f}%** on `data/v29_clean_generalisation_tasks.jsonl`",
            "",
            "This test set contains tasks similar to the 4 repair targets but with",
            "different names and examples — never seen during training or memory repair.",
            "",
            "| Task | Result |",
            "|------|--------|",
        ]
        clean_map = _task_pass_map(clean_rows)
        for task, passed in clean_map.items():
            lines.append(f"| {task} | {'PASS' if passed else 'FAIL'} |")
        lines += [
            "",
            f"Generalisation claim: repair memory {'generalises' if clean_pct >= 75 else 'does not clearly generalise'} "
            f"to similar but unseen tasks ({clean_pass}/{clean_n} = {clean_pct:.1f}%).",
        ]

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[summarise_v29] Written: {output_dir / 'summary.md'}", file=sys.stderr)

    # ── claim_boundary.md ───────────────────────────────────────────────────
    claim_lines = [
        "# AetherForge v2.9 — Claim Boundary",
        f"\nGenerated: {now}\n",
        "## What v2.9 can claim",
        "",
        "1. **Retrieval noise identified**: four failing tasks (`merge_intervals`,",
        "   `median_two_sorted`, `deep_get`, `tree_depth_tuple`) all receive",
        "   misleading context from the champion index at k=4.",
        "   - `merge_intervals` and `median_two_sorted`: dominated by `merge_sorted` records",
        "     due to surface lexical similarity ('merge', 'two sorted').",
        "   - `tree_depth_tuple`: dominated by `flatten` records (nested-list pattern).",
        "   - `deep_get`: dominated by `invert_dict` + `flatten` records.",
        "",
        "2. **Broken task assertion discovered**: `tree_depth_tuple` task prompt",
        "   asserts `tree_depth(((1,2),(3,(4,5)))) == 3` but the correct value",
        "   by the stated rule (leaves=depth 1, branch=1+max) is **4**.",
        "   This broken assertion may cause correct implementations to fail if the",
        "   model copies the task's assertion verbatim into its test code.",
        "",
        "3. **Diagnostic repair** (same benchmark + repair records):",
        "   evaluation on the original 28-task benchmark after adding repair examples.",
        "   NOT a clean promotion regardless of score.",
        "",
        "4. **Generalisation result** (clean test set):",
        "   score on `data/v29_clean_generalisation_tasks.jsonl` is a clean result",
        "   for the claim that repair memory generalises to similar unseen tasks.",
        "",
        "## What v2.9 cannot claim",
        "",
        "- A new 28-task champion based on diagnostic repair results alone.",
        "- Any improvement to the original 23/28 = 82.1% score",
        "  (the champion index is untouched in v2.9).",
        "- That the repair approach is contamination-free on the 28-task benchmark.",
        "",
        "## Path to a clean 28-task promotion",
        "",
        "1. Merge `memory/raw_v29_repair/repair_records.jsonl` into `memory/raw_adapted/`.",
        "2. Rebuild `memory/index_adapted` from the merged raw directory.",
        "3. Re-run the full 28-task benchmark with the updated champion index.",
        "4. If score >= 24/28: new champion at that score.",
        "5. If score == 23/28: tie, no promotion, but memory augmentation is neutral.",
        "6. The generalisation result from step 4 of v2.9 is still valid.",
    ]
    (output_dir / "claim_boundary.md").write_text("\n".join(claim_lines) + "\n")
    print(f"[summarise_v29] Written: {output_dir / 'claim_boundary.md'}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise v2.9 memory repair results")
    p.add_argument("--diagnostic-csv",
                   default="results/v29_memory_repair/diagnostic_repair_results.csv")
    p.add_argument("--clean-csv",
                   default="results/v29_memory_repair/clean_generalisation_results.csv")
    p.add_argument("--output-dir",
                   default="results/v29_memory_repair")
    args = p.parse_args()

    output_dir     = Path(args.output_dir)
    diagnostic_rows = _read_csv(Path(args.diagnostic_csv))
    clean_rows      = _read_csv(Path(args.clean_csv))

    build_summary(diagnostic_rows, clean_rows, output_dir)
    print(f"Done. See {output_dir}/summary.md and {output_dir}/claim_boundary.md")


if __name__ == "__main__":
    main()
