#!/usr/bin/env python3
"""
scripts/summarise_v29_promotion.py
AetherForge v2.9 — Repair-Index Promotion Summary

A/B comparison of:
  Lane 1: memory/index_adapted    (99 records, clean champion)
  Lane 2: memory/index_adapted_v29 (103 records, repair-enhanced)

Produces:
  results/v29_memory_repair/promotion_summary.md
  results/v29_memory_repair/index_adapted_v29_eval.md
  results/v29_memory_repair/clean_generalisation_eval.md

Claim boundary:
  Lane 2 result on the 28-task frozen benchmark = repair-index diagnostic.
  NOT a clean champion promotion (repair records target known benchmark failures).
  Clean generalisation result is a valid external signal.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BASELINE_SCORE = 23
BASELINE_TOTAL = 28
BASELINE_PCT   = 100.0 * BASELINE_SCORE / BASELINE_TOTAL
PROMOTION_THRESHOLD = 24
SPEC_CONFLICTED = {"tree_depth_tuple"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _pass_count(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("passed", "").strip().lower() in ("true", "1", "yes", "pass"))


def _task_map(rows: list[dict]) -> dict[str, bool]:
    return {r["task_id"]: r.get("passed","").strip().lower() in ("true","1","yes","pass")
            for r in rows if "task_id" in r}


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise v2.9 repair-index promotion")
    p.add_argument("--adapted-v29-28task-csv",
                   default="results/v29_memory_repair/adapted_v29_28task_results.csv")
    p.add_argument("--adapted-v29-clean-csv",
                   default="results/v29_memory_repair/adapted_v29_clean_results.csv")
    p.add_argument("--output-dir", default="results/v29_memory_repair")
    args = p.parse_args()

    output_dir   = Path(args.output_dir)
    rows_28task  = _read_csv(Path(args.adapted_v29_28task_csv))
    rows_clean   = _read_csv(Path(args.adapted_v29_clean_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    n28   = len(rows_28task)
    p28   = _pass_count(rows_28task)
    pct28 = 100.0 * p28 / n28 if n28 else 0.0
    map28 = _task_map(rows_28task)

    # Corrected audit excluding spec-conflicted tasks
    rows_corr = [r for r in rows_28task if r.get("task_id") not in SPEC_CONFLICTED]
    n_corr    = len(rows_corr)
    p_corr    = _pass_count(rows_corr)
    pct_corr  = 100.0 * p_corr / n_corr if n_corr else 0.0

    n_clean   = len(rows_clean)
    p_clean   = _pass_count(rows_clean)
    pct_clean = 100.0 * p_clean / n_clean if n_clean else 0.0
    map_clean = _task_map(rows_clean)

    # ── promotion_summary.md ────────────────────────────────────────────────
    champion_tasks = {
        "merge_intervals": "FAIL", "median_two_sorted": "FAIL",
        "deep_get": "FAIL", "tree_depth_tuple": "FAIL",
        "rle_decode": "FAIL",
    }

    lines = [
        "# AetherForge v2.9 — Repair-Index Promotion Summary",
        f"\nGenerated: {now}\n",
        "## A/B Comparison",
        "",
        "| Lane | Index | Score | Notes |",
        "|------|-------|-------|-------|",
        f"| 1 (clean champion) | `memory/index_adapted` | {BASELINE_SCORE}/{BASELINE_TOTAL} = {BASELINE_PCT:.1f}% | Original champion, untouched |",
    ]
    if rows_28task:
        delta_str = f"+{p28 - BASELINE_SCORE}" if p28 >= BASELINE_SCORE else str(p28 - BASELINE_SCORE)
        lines.append(
            f"| 2 (repair-enhanced) | `memory/index_adapted_v29` | {p28}/{n28} = {pct28:.1f}% "
            f"({delta_str} vs champion) | REPAIR-INDEX DIAGNOSTIC — NOT clean champion |"
        )
        if n_corr < n28:
            lines.append(
                f"| 2 corrected | (excl. spec-conflicted) | {p_corr}/{n_corr} = {pct_corr:.1f}% "
                f"| Excludes `tree_depth_tuple` |"
            )
    else:
        lines.append("| 2 (repair-enhanced) | `memory/index_adapted_v29` | _pending_ | |")
    if rows_clean:
        lines.append(
            f"| 2 clean gen | clean task set | {p_clean}/{n_clean} = {pct_clean:.1f}% "
            f"| Valid generalisation signal |"
        )

    lines += ["", "## Claim boundary", ""]

    if rows_28task:
        if p28 >= PROMOTION_THRESHOLD:
            lines += [
                f"**Lane 2 score {p28}/{n28} = {pct28:.1f}% is above the promotion threshold "
                f"({PROMOTION_THRESHOLD}/{n28}).**",
                "",
                "⚠️  This result is a **repair-index diagnostic**, not a clean held-out champion.",
                "The index was augmented with repair records targeting the known failing tasks.",
                "The frozen 28-task benchmark is no longer fully independent for Lane 2.",
                "",
                "To promote a clean champion, create a new untouched test set, OR commit to the",
                "repair-enhanced index as the production index and report honestly:",
                f"  - Clean champion (Lane 1): {BASELINE_SCORE}/{BASELINE_TOTAL} = {BASELINE_PCT:.1f}%",
                f"  - Repair-index diagnostic (Lane 2): {p28}/{n28} = {pct28:.1f}%",
            ]
        else:
            lines += [
                f"Lane 2 score {p28}/{n28} = {pct28:.1f}% does not meet the promotion threshold.",
                "The repair-enhanced index shows limited improvement on the diagnostic run.",
            ]
    else:
        lines.append("_Pending: run `make eval-v29-adapted-repair-index`_")

    lines += [
        "",
        "## Per-task outcomes (repair targets)",
        "",
        "| Task | Lane 1 (champion) | Lane 2 (repair-enhanced) | Type |",
        "|------|-------------------|--------------------------|------|",
    ]
    repair_targets = {
        "merge_intervals":   ("FAIL", "valid_repair"),
        "median_two_sorted": ("FAIL", "valid_repair"),
        "deep_get":          ("FAIL", "valid_repair"),
        "tree_depth_tuple":  ("FAIL", "benchmark_defect_repair"),
    }
    for task, (l1, rtype) in repair_targets.items():
        if map28:
            l2 = "**PASS**" if map28.get(task, False) else "FAIL"
        else:
            l2 = "_pending_"
        tag = "spec-conflicted" if task in SPEC_CONFLICTED else rtype.replace("_", " ")
        lines.append(f"| `{task}` | {l1} | {l2} | {tag} |")

    lines += [
        "",
        "**tree_depth_tuple** is marked `spec-conflicted` because the task prompt asserts",
        "`tree_depth(((1,2),(3,(4,5)))) == 3` but the correct value by the stated rule is 4.",
        "A PASS on this task with repair memory means the model followed the correct repair",
        "example rather than the task's broken assertion.",
    ]

    (output_dir / "promotion_summary.md").write_text("\n".join(lines) + "\n")

    # ── index_adapted_v29_eval.md ───────────────────────────────────────────
    if rows_28task:
        eval_lines = [
            "# index_adapted_v29 — Full 28-task Eval",
            f"\nGenerated: {now}",
            f"\nRaw frozen: {p28}/{n28} = {pct28:.1f}%",
            f"Corrected audit (excl. tree_depth_tuple): {p_corr}/{n_corr} = {pct_corr:.1f}%\n",
            "| Task | Category | Result |",
            "|------|----------|--------|",
        ]
        for r in rows_28task:
            passed = r.get("passed","").lower() in ("true","1","yes","pass")
            note = " *(spec-conflicted)*" if r["task_id"] in SPEC_CONFLICTED else ""
            eval_lines.append(f"| `{r['task_id']}` | {r.get('category','')} | {'PASS' if passed else 'FAIL'}{note} |")
        (output_dir / "index_adapted_v29_eval.md").write_text("\n".join(eval_lines) + "\n")

    # ── clean_generalisation_eval.md ───────────────────────────────────────
    if rows_clean:
        clean_lines = [
            "# Repair-index Clean Generalisation Eval",
            f"\nGenerated: {now}",
            f"\nScore: {p_clean}/{n_clean} = {pct_clean:.1f}%\n",
            "| Task | Result | Pattern |",
            "|------|--------|---------|",
        ]
        pattern_map = {
            "v29_clean_merge_time_slots": "sort-and-sweep (merge_intervals variant)",
            "v29_clean_find_kth_sorted":  "two-sorted k-th stop (median_two_sorted variant)",
            "v29_clean_deep_set":         "nested-dict traversal (deep_get variant)",
            "v29_clean_tree_node_count":  "recursive tuple-tree (tree_depth_tuple variant)",
            "v29_clean_merge_k_sorted":   "pairwise two-sorted merge (median_two_sorted variant)",
        }
        for r in rows_clean:
            passed = r.get("passed","").lower() in ("true","1","yes","pass")
            pat = pattern_map.get(r["task_id"], "")
            clean_lines.append(f"| `{r['task_id']}` | {'PASS' if passed else 'FAIL'} | {pat} |")
        (output_dir / "clean_generalisation_eval.md").write_text("\n".join(clean_lines) + "\n")

    print(f"[summarise_v29_promotion] Written to {output_dir}")
    print(f"  promotion_summary.md")
    if rows_28task:
        print(f"  index_adapted_v29_eval.md")
    if rows_clean:
        print(f"  clean_generalisation_eval.md")


if __name__ == "__main__":
    main()
