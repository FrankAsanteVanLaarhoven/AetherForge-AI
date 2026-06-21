#!/usr/bin/env python3
"""
scripts/summarise_v28_champion_system.py
AetherForge v2.8 — Champion System Enhancement

Compare all v2.8 eval configurations against the 23/28 merged-champion baseline.
Produces:
  results/v28_champion_system/summary.md
  results/v28_champion_system/per_task_comparison.csv
  results/v28_champion_system/failure_analysis.md

Promotion rule:
  >= 24/28  → new champion
  == 23/28  → tie (no promotion)
  <  23/28  → reject

Failure categories (applied to the 5 tasks that fail in the current champion):
  wrong_retrieval       memory returned an irrelevant example
  missing_retrieval     no memory hit; the model had no context to anchor on
  prompt_misunderstanding  the model mis-read the task specification
  incomplete_reasoning  partial / truncated solution
  wrong_tool_use        incorrect tool call structure or tool name
  answer_format_error   correct logic, wrong output format
  generation_issue      decoding / repetition / truncation artefact
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Known failing tasks in current best system (merged champion + memory, 23/28)
# Used to annotate the failure analysis.
# ---------------------------------------------------------------------------

CURRENT_CHAMPION_FAILS = {
    "group_anagrams":   "medium",
    "merge_intervals":  "hard",
    "count_islands":    "hard",
    "median_two_sorted": "hard",
    "tree_depth_tuple": "hard",
}

PROMOTION_THRESHOLD = 24
CHAMPION_SCORE      = 23
TOTAL_TASKS         = 28

FAILURE_CATEGORIES = [
    "wrong_retrieval",
    "missing_retrieval",
    "prompt_misunderstanding",
    "incomplete_reasoning",
    "wrong_tool_use",
    "answer_format_error",
    "generation_issue",
]

# Config key → human label for the summary table
LABELS = {
    "current_champion":    "Merged champion + memory (baseline)",
    "no_memory":           "Merged champion — memory DISABLED",
    "memory_topk1":        "Memory top-k = 1",
    "memory_topk3":        "Memory top-k = 3",
    "memory_topk5":        "Memory top-k = 5",
    "filtered_memory":     "Filtered memory index",
    "direct_answer_prompt": "Direct-answer system prompt",
    "continuation_logic":  "Continuation logic (best-of-5 on failing tasks)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _passed(row: dict) -> bool:
    return row.get("passed", "").strip().lower() in ("true", "1", "yes")


def _pass_rate(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, 0.0
    n = len(rows)
    p = sum(1 for r in rows if _passed(r))
    return p, n, round(100 * p / n, 1)


def _by_category(rows: list[dict]) -> dict[str, tuple[int, int]]:
    cats: dict[str, list] = defaultdict(list)
    for r in rows:
        cats[r.get("category", "unknown")].append(_passed(r))
    return {c: (sum(v), len(v)) for c, v in cats.items()}


def _promotion_verdict(passed: int, total: int) -> str:
    if total == 0:
        return "—"
    if passed >= PROMOTION_THRESHOLD:
        return "PROMOTE"
    if passed == CHAMPION_SCORE:
        return "tie"
    return "reject"


# ---------------------------------------------------------------------------
# Failure classifier
# Based on per-task CSV output from evaluate_code_agent.
# Reads extended columns if available (has_invalid_json, steps, tool_calls …).
# Falls back to heuristic classification when those columns are absent.
# ---------------------------------------------------------------------------

def _classify_failure(row: dict) -> str:
    """Classify a failing task row into one of the FAILURE_CATEGORIES."""
    if _passed(row):
        return ""

    # Extended columns from evaluate_code_agent (present when --verbose)
    has_invalid_json  = row.get("has_invalid_json", "").strip().lower() in ("true", "1")
    has_unknown_tool  = row.get("has_unknown_tool", "").strip().lower() in ("true", "1")
    has_direct_task   = row.get("has_direct_task_tool_call", "").strip().lower() in ("true", "1")
    used_fallback     = row.get("used_fallback_extraction", "").strip().lower() in ("true", "1")
    failure_reason    = row.get("failure_reason", "").strip().lower()
    steps_str         = row.get("steps", "0")
    tool_calls_str    = row.get("tool_calls", "0")

    try:
        steps = int(steps_str)
    except ValueError:
        steps = 0
    try:
        tool_calls = int(tool_calls_str)
    except ValueError:
        tool_calls = 0

    if has_invalid_json or has_unknown_tool:
        return "wrong_tool_use"
    if has_direct_task:
        return "wrong_tool_use"
    if "format" in failure_reason or used_fallback:
        return "answer_format_error"
    if steps <= 1 and tool_calls == 0:
        return "generation_issue"
    if steps == 1 and tool_calls > 0:
        return "incomplete_reasoning"
    if "misunderstand" in failure_reason or "wrong task" in failure_reason:
        return "prompt_misunderstanding"
    return "incomplete_reasoning"


# ---------------------------------------------------------------------------
# Per-task table
# ---------------------------------------------------------------------------

def _build_task_table(results: dict[str, list[dict] | None]) -> list[dict]:
    task_ids: list[str] = []
    task_meta: dict[str, dict] = {}
    seen: set[str] = set()
    for rows in results.values():
        if rows is None:
            continue
        for r in rows:
            tid = r.get("id", "")
            if tid and tid not in seen:
                task_ids.append(tid)
                task_meta[tid] = {
                    "category": r.get("category", ""),
                    "is_current_fail": tid in CURRENT_CHAMPION_FAILS,
                }
                seen.add(tid)

    table = []
    for tid in task_ids:
        row: dict = {
            "id":        tid,
            "category":  task_meta[tid]["category"],
            "target":    "YES" if task_meta[tid]["is_current_fail"] else "",
        }
        for label, rows in results.items():
            if rows is None:
                row[label] = "—"
            else:
                match = [r for r in rows if r.get("id") == tid]
                if match:
                    row[label] = "PASS" if _passed(match[0]) else "FAIL"
                else:
                    row[label] = "—"
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_summary(results: dict, output_dir: Path) -> Path:
    path = output_dir / "summary.md"
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# AetherForge v2.8 — Champion System Enhancement",
        "",
        f"Generated: {ts}",
        "",
        "## Goal",
        "",
        "Improve the 82.1% merged champion + memory system to >= 85.7% (24/28)",
        "without retraining the model.",
        "",
        "## Known baseline (v2.7 merged champion + memory)",
        "",
        "| Task | Category | Status |",
        "|---|---|---|",
    ]
    for tid, cat in CURRENT_CHAMPION_FAILS.items():
        lines.append(f"| {tid} | {cat} | FAIL |")
    lines += [
        "",
        "## v2.8 Experiment results",
        "",
        "| Configuration | Score | pp vs baseline | Verdict |",
        "|---|---|---|---|",
    ]

    baseline_rows = results.get("current_champion")
    baseline_p    = CHAMPION_SCORE
    if baseline_rows is not None:
        baseline_p, _, _ = _pass_rate(baseline_rows)

    for key, label in LABELS.items():
        rows = results.get(key)
        if rows is None:
            lines.append(f"| {label} | *pending* | — | — |")
        else:
            p, n, rate = _pass_rate(rows)
            delta = p - baseline_p
            delta_str = f"{delta:+d}"
            verdict = _promotion_verdict(p, n)
            lines.append(f"| {label} | {p}/{n} = {rate:.1f}% | {delta_str} | {verdict} |")

    lines += [
        "",
        "## Promotion rule",
        "",
        f"- **24/28 or better** → new champion",
        f"- **23/28** → tie, no promotion",
        f"- **< 23/28** → reject",
        "",
        "## Memory sensitivity (top-k ablation)",
        "",
        "| top-k | Score | pp vs k=4 |",
        "|---|---|---|",
    ]
    k4_rows = results.get("current_champion")
    k4_p    = CHAMPION_SCORE
    if k4_rows is not None:
        k4_p, _, _ = _pass_rate(k4_rows)
    for key, label in [
        ("memory_topk1", "k = 1"),
        ("current_champion", "k = 4  (baseline)"),
        ("memory_topk3", "k = 3"),
        ("memory_topk5", "k = 5"),
    ]:
        rows = results.get(key)
        if rows is None:
            lines.append(f"| {label} | *pending* | — |")
        else:
            p, n, _ = _pass_rate(rows)
            lines.append(f"| {label} | {p}/{n} | {p - k4_p:+d} |")

    lines += [
        "",
        "## Claim boundary",
        "",
        "- Current system: merged champion + memory/index_adapted = 23/28 = 82.1%",
        "- Do NOT claim retraining improvement from v2.8 targets.",
        "- Do NOT claim SWE-bench success, production-grade agent, or AGI.",
        "- Champion path: `outputs/qwen15b_v27_champion_merged`",
        "- Memory index: `memory/index_adapted`",
        "",
    ]

    path.write_text("\n".join(lines))
    return path


def _write_per_task_csv(table: list[dict], output_dir: Path) -> Path:
    path = output_dir / "per_task_comparison.csv"
    if not table:
        path.write_text("id,category,target\n")
        return path
    fields = list(table[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    return path


def _write_failure_analysis(results: dict, output_dir: Path) -> Path:
    path = output_dir / "failure_analysis.md"
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# v2.8 Failure Analysis",
        "",
        f"Generated: {ts}",
        "",
        "Tasks that fail in the current 23/28 champion (merged + memory):",
        "",
    ]
    for tid, cat in CURRENT_CHAMPION_FAILS.items():
        lines.append(f"- **{tid}** ({cat})")
    lines.append("")

    # For each config, show which of the target tasks passed/failed and classify
    for key, label in LABELS.items():
        rows = results.get(key)
        lines.append(f"## {label}")
        if rows is None:
            lines.append("*Not yet evaluated*\n")
            continue

        target_rows = [r for r in rows if r.get("id") in CURRENT_CHAMPION_FAILS]
        if not target_rows:
            lines.append("*Target tasks not found in this eval output*\n")
            continue

        p, n, rate = _pass_rate(target_rows)
        lines.append(f"Target task pass rate: {p}/{n}")
        lines.append("")
        lines.append("| Task | Category | Result | Failure class |")
        lines.append("|---|---|---|---|")
        for r in target_rows:
            tid     = r.get("id", "")
            cat     = r.get("category", "")
            verdict = "PASS" if _passed(r) else "FAIL"
            fclass  = _classify_failure(r) if not _passed(r) else ""
            lines.append(f"| {tid} | {cat} | {verdict} | {fclass} |")
        lines.append("")

    lines += [
        "## Failure category reference",
        "",
    ]
    for fc in FAILURE_CATEGORIES:
        lines.append(f"- **{fc}**")

    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="v2.8 Champion System Enhancement summary"
    )
    ap.add_argument("--output-dir",
                    default="results/v28_champion_system")
    ap.add_argument("--current-champion-csv",
                    default="outputs/eval_v28_current_champion/best_of_3.csv")
    ap.add_argument("--no-memory-csv",
                    default="outputs/eval_v28_no_memory/best_of_3.csv")
    ap.add_argument("--topk1-csv",
                    default="outputs/eval_v28_memory_topk1/best_of_3.csv")
    ap.add_argument("--topk3-csv",
                    default="outputs/eval_v28_memory_topk3/best_of_3.csv")
    ap.add_argument("--topk5-csv",
                    default="outputs/eval_v28_memory_topk5/best_of_3.csv")
    ap.add_argument("--filtered-memory-csv",
                    default="outputs/eval_v28_filtered_memory/best_of_3.csv")
    ap.add_argument("--direct-answer-csv",
                    default="outputs/eval_v28_direct_answer_prompt/best_of_3.csv")
    ap.add_argument("--continuation-csv",
                    default="outputs/eval_v28_continuation_logic/best_of_5.csv")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict] | None] = {
        "current_champion":    _load_csv(Path(args.current_champion_csv)),
        "no_memory":           _load_csv(Path(args.no_memory_csv)),
        "memory_topk1":        _load_csv(Path(args.topk1_csv)),
        "memory_topk3":        _load_csv(Path(args.topk3_csv)),
        "memory_topk5":        _load_csv(Path(args.topk5_csv)),
        "filtered_memory":     _load_csv(Path(args.filtered_memory_csv)),
        "direct_answer_prompt": _load_csv(Path(args.direct_answer_csv)),
        "continuation_logic":  _load_csv(Path(args.continuation_csv)),
    }

    available = [k for k, v in results.items() if v is not None]
    missing   = [k for k, v in results.items() if v is None]
    print(f"\nv2.8 Champion System Enhancement — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Available : {available}")
    print(f"Missing   : {missing}")
    print()

    for key, rows in results.items():
        if rows is not None:
            p, n, rate = _pass_rate(rows)
            label = LABELS.get(key, key)
            delta = p - CHAMPION_SCORE
            verdict = _promotion_verdict(p, n)
            cats = _by_category(rows)
            print(f"  {key:25s}: {p}/{n} = {rate:.1f}%  ({delta:+d} pp)  [{verdict}]")
            for cat, (cp, cn) in sorted(cats.items()):
                print(f"    {cat:12s}: {cp}/{cn} = {100*cp//cn if cn else 0}%")
        else:
            print(f"  {key:25s}: not yet evaluated")

    print()
    print("Target tasks (currently failing in 23/28 champion):")
    for key, rows in results.items():
        if rows is None:
            continue
        target = [r for r in rows if r.get("id") in CURRENT_CHAMPION_FAILS]
        if target:
            p = sum(1 for r in target if _passed(r))
            print(f"  {key:25s}: {p}/{len(target)} target tasks pass")

    task_table   = _build_task_table(results)
    p1 = _write_summary(results, out)
    p2 = _write_per_task_csv(task_table, out)
    p3 = _write_failure_analysis(results, out)

    print(f"\nOutputs written:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")


if __name__ == "__main__":
    main()
