#!/usr/bin/env python3
"""
scripts/inspect_v29_retrieval.py
AetherForge v2.9 — Memory Repair Split: retrieval inspection

For each of the 4 failing tasks, retrieve top-k=4 from both the champion index
and the repair index (champion + repair records), label hits as HELPFUL or HARMFUL,
and write a markdown report.

Usage:
    conda run -n ml-torch python scripts/inspect_v29_retrieval.py \
        --champion-index memory/index_adapted \
        --repair-raw-dir memory/raw_v29_repair \
        --repair-index   memory/index_v29_repair \
        --output-md      results/v29_memory_repair/retrieval_inspection.md

The repair index is NOT memory/index_adapted — it is built from
memory/raw_adapted + memory/raw_v29_repair combined.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.core import retrieve, build_index

# ── Target tasks and their known query strings ────────────────────────────

TARGET_TASKS = {
    "merge_intervals": {
        "query": (
            "hard merge_intervals Write merge_intervals(intervals) where "
            "intervals is a list of [start, end] pairs. Sort by start, merge "
            "overlapping or adjacent intervals, return sorted merged list."
        ),
        "harmful_patterns": ["merge_sorted", "merge two sorted", "merge two lists"],
        "helpful_patterns": ["merge_intervals", "interval", "overlap", "sweep"],
    },
    "median_two_sorted": {
        "query": (
            "hard median_two_sorted Write median_of_two(a, b) returning the "
            "median of two sorted arrays combined. Binary search or linear merge."
        ),
        "harmful_patterns": ["merge_sorted", "merge two sorted", "merge two lists"],
        "helpful_patterns": ["median", "two sorted", "kth", "binary search"],
    },
    "deep_get": {
        "query": (
            "hard deep_get Write deep_get(d, keys, default=None) navigating a "
            "nested dict by a list of keys, returning value or default if any "
            "key is missing or intermediate value is not a dict."
        ),
        "harmful_patterns": ["invert_dict", "flatten", "flatten list"],
        "helpful_patterns": ["deep_get", "nested dict", "key path", "traverse"],
    },
    "tree_depth_tuple": {
        "query": (
            "hard tree_depth_tuple Write tree_depth(node) where a tree is a "
            "nested tuple: leaf is an integer, branch is (left, right). "
            "Return maximum depth where leaves have depth 1."
        ),
        "harmful_patterns": ["flatten", "flatten list", "nested list"],
        "helpful_patterns": ["tree_depth", "depth", "recursive", "tuple tree"],
    },
}


def label_record(rec: dict, task_info: dict) -> str:
    """Return 'HELPFUL', 'HARMFUL', or 'NEUTRAL' for a retrieved record."""
    task_text = rec.get("task", "").lower()
    query_text = rec.get("query_text", "").lower()
    combined = task_text + " " + query_text

    for pat in task_info["harmful_patterns"]:
        if pat.lower() in combined:
            return "HARMFUL"
    for pat in task_info["helpful_patterns"]:
        if pat.lower() in combined:
            return "HELPFUL"
    return "NEUTRAL"


def inspect_index(task_name: str, task_info: dict, index_dir: Path, top_k: int = 4) -> list[dict]:
    """Retrieve top-k hits for a task from an index and label them."""
    hits = retrieve(task_text=task_info["query"], index_dir=index_dir, top_k=top_k)
    results = []
    for rank, h in enumerate(hits, 1):
        results.append({
            "rank": rank,
            "score": h.get("score", 0.0),
            "id": h.get("id", "")[:8],
            "category": h.get("category", ""),
            "task_preview": h.get("task", "")[:80],
            "label": label_record(h, task_info),
        })
    return results


def format_table(hits: list[dict]) -> str:
    lines = ["| Rank | Score | ID | Category | Label | Task (preview) |",
             "|------|-------|----|----------|-------|----------------|"]
    for h in hits:
        lines.append(
            f"| {h['rank']} | {h['score']:.4f} | {h['id']} "
            f"| {h['category']} | **{h['label']}** | {h['task_preview']} |"
        )
    return "\n".join(lines)


def build_repair_index_if_needed(repair_raw_dir: Path, repair_index_dir: Path) -> None:
    """Merge raw_adapted + raw_v29_repair into a combined raw dir and build index."""
    import shutil, tempfile

    combined_raw = Path("memory") / "_tmp_v29_combined_raw"
    combined_raw.mkdir(exist_ok=True)

    # Copy all files from raw_adapted
    raw_adapted = Path("memory/raw_adapted")
    if raw_adapted.exists():
        for f in raw_adapted.glob("*.jsonl"):
            dst = combined_raw / f.name
            if not dst.exists():
                shutil.copy(f, dst)

    # Copy repair records (separate filename to avoid collision)
    for f in repair_raw_dir.glob("*.jsonl"):
        dst = combined_raw / f"v29_repair_{f.name}"
        shutil.copy(f, dst)

    build_index(raw_dir=combined_raw, index_dir=repair_index_dir)
    # Clean up temp dir
    shutil.rmtree(combined_raw)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect v2.9 retrieval for failing tasks")
    p.add_argument("--champion-index", default="memory/index_adapted",
                   help="Champion memory index directory")
    p.add_argument("--repair-raw-dir", default="memory/raw_v29_repair",
                   help="Raw directory with v2.9 repair records")
    p.add_argument("--repair-index", default="memory/index_v29_repair",
                   help="Repair index directory (champion + repair records)")
    p.add_argument("--output-md", default="results/v29_memory_repair/retrieval_inspection.md",
                   help="Path for the output markdown report")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--rebuild-repair-index", action="store_true",
                   help="Rebuild the repair index even if it already exists")
    args = p.parse_args()

    champion_index = Path(args.champion_index)
    repair_raw_dir = Path(args.repair_raw_dir)
    repair_index   = Path(args.repair_index)
    output_md      = Path(args.output_md)

    if not champion_index.exists():
        print(f"ERROR: champion index not found at {champion_index}", file=sys.stderr)
        sys.exit(1)

    if not repair_raw_dir.exists():
        print(f"ERROR: repair raw dir not found at {repair_raw_dir}", file=sys.stderr)
        sys.exit(1)

    if not repair_index.exists() or args.rebuild_repair_index:
        print(f"[inspect_v29] Building repair index at {repair_index} ...", file=sys.stderr)
        build_repair_index_if_needed(repair_raw_dir, repair_index)
        print(f"[inspect_v29] Repair index built.", file=sys.stderr)

    output_md.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    sections.append("# v2.9 Retrieval Inspection\n")
    sections.append(
        "Comparing top-k=4 hits from the champion index vs the repair index "
        "(champion + 4 task-specific repair records) for the 4 failing tasks.\n"
    )
    sections.append(
        "Labels: **HELPFUL** = pattern relevant to the task, "
        "**HARMFUL** = wrong algorithm injected, **NEUTRAL** = unrelated.\n"
    )

    summary_rows = []

    for task_name, task_info in TARGET_TASKS.items():
        print(f"[inspect_v29] Inspecting {task_name} ...", file=sys.stderr)

        champ_hits   = inspect_index(task_name, task_info, champion_index, args.top_k)
        repair_hits  = inspect_index(task_name, task_info, repair_index, args.top_k)

        champ_harmful  = sum(1 for h in champ_hits  if h["label"] == "HARMFUL")
        champ_helpful  = sum(1 for h in champ_hits  if h["label"] == "HELPFUL")
        repair_harmful = sum(1 for h in repair_hits if h["label"] == "HARMFUL")
        repair_helpful = sum(1 for h in repair_hits if h["label"] == "HELPFUL")

        summary_rows.append({
            "task": task_name,
            "champ_harmful": champ_harmful,
            "champ_helpful": champ_helpful,
            "repair_harmful": repair_harmful,
            "repair_helpful": repair_helpful,
        })

        sections.append(f"\n## {task_name}\n")
        sections.append(f"### Champion index (k={args.top_k})\n")
        sections.append(format_table(champ_hits))
        sections.append(f"\n### Repair index (k={args.top_k})\n")
        sections.append(format_table(repair_hits))

    # Summary table
    sections.insert(3, "\n## Summary\n")
    summary_lines = [
        "| Task | Champion HARMFUL | Champion HELPFUL | Repair HARMFUL | Repair HELPFUL |",
        "|------|-----------------|-----------------|----------------|----------------|",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| {row['task']} | {row['champ_harmful']} | {row['champ_helpful']} "
            f"| {row['repair_harmful']} | {row['repair_helpful']} |"
        )
    sections.insert(4, "\n".join(summary_lines) + "\n")

    output_md.write_text("\n".join(sections))
    print(f"[inspect_v29] Report written to {output_md}", file=sys.stderr)
    print(output_md)


if __name__ == "__main__":
    main()
