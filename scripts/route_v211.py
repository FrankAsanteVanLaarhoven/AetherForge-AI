#!/usr/bin/env python3
"""
scripts/route_v211.py
AetherForge v2.11 — Retrieval Routing and Gating Audit

Queries both champion and repair indexes for each task, applies routing rules,
and splits tasks into sub-files that can be evaluated with the correct index.

Outputs (all under --output-dir):
  routing_scores.csv          — per-task retrieval scores from both indexes
  routing_decisions.json      — {task_id: {family_router, confidence_router, oracle_router}}
  tasks_family_champion.jsonl — tasks routed to champion by family router
  tasks_family_repair.jsonl   — tasks routed to repair by family router
  tasks_conf_champion.jsonl   — tasks routed to champion by confidence router
  tasks_conf_repair.jsonl     — tasks routed to repair by confidence router

Oracle routing is computed from v2.10 CSV results — no new eval needed.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.core import retrieve

INTERVAL_MERGE_FAMILY = "interval_merge"


# ── Routing rules ─────────────────────────────────────────────────────────────

def family_route(family: str) -> str:
    return "repair" if family == INTERVAL_MERGE_FAMILY else "champion"


def confidence_route(
    champion_top1: float,
    repair_top1: float,
    threshold: float = 0.35,
    margin: float = 0.05,
) -> str:
    if repair_top1 >= threshold and (repair_top1 - champion_top1) >= margin:
        return "repair"
    return "champion"


def oracle_route(champion_passed: bool, repair_passed: bool) -> str:
    """Choose whichever index passes — diagnostic ceiling, not deployable."""
    if repair_passed and not champion_passed:
        return "repair"
    return "champion"


# ── Retrieval scoring ─────────────────────────────────────────────────────────

def get_top1_score(task_text: str, index_dir: Path) -> float:
    hits = retrieve(task_text, index_dir, top_k=1, min_score=0.0)
    return hits[0].get("score", 0.0) if hits else 0.0


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_v210_passed(csv_path: Path) -> dict[str, bool]:
    if not csv_path.exists():
        print(f"[route_v211] WARNING: {csv_path} not found — oracle routing unavailable",
              file=sys.stderr)
        return {}
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tid = row.get("id", row.get("task_id", "")).strip()
            passed = row.get("passed", "").strip().lower() in ("true", "1", "yes", "pass")
            if tid:
                result[tid] = passed
    return result


def write_jsonl(tasks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"[route_v211] Written {len(tasks)} tasks → {path}", file=sys.stderr)


def write_routing_scores_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[route_v211] Written routing scores → {path}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compute routing decisions for v2.11")
    p.add_argument("--tasks-file",
                   default="data/v210_clean_repair_generalisation_tasks.jsonl")
    p.add_argument("--champion-index", default="memory/index_adapted")
    p.add_argument("--repair-index",   default="memory/index_adapted_v29")
    p.add_argument("--champion-v210-csv",
                   default="results/v210_clean_repair_generalisation/champion_results.csv")
    p.add_argument("--repair-v210-csv",
                   default="results/v210_clean_repair_generalisation/repair_results.csv")
    p.add_argument("--output-dir",
                   default="results/v211_retrieval_routing")
    p.add_argument("--confidence-threshold", type=float, default=0.35,
                   help="Repair top-1 score must exceed this to trigger repair routing")
    p.add_argument("--margin-threshold", type=float, default=0.05,
                   help="Repair score must exceed champion by at least this margin")
    args = p.parse_args()

    tasks_file     = Path(args.tasks_file)
    champion_index = Path(args.champion_index)
    repair_index   = Path(args.repair_index)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tasks_file.exists():
        print(f"ERROR: tasks file not found: {tasks_file}", file=sys.stderr)
        sys.exit(1)
    if not champion_index.exists():
        print(f"ERROR: champion index not found: {champion_index}", file=sys.stderr)
        sys.exit(1)
    if not repair_index.exists():
        print(f"ERROR: repair index not found: {repair_index}", file=sys.stderr)
        sys.exit(1)

    tasks = [json.loads(l) for l in tasks_file.open()]
    champion_v210 = read_v210_passed(Path(args.champion_v210_csv))
    repair_v210   = read_v210_passed(Path(args.repair_v210_csv))

    score_rows     = []
    decisions      = {}
    fam_champion   = []
    fam_repair     = []
    conf_champion  = []
    conf_repair    = []

    print(f"[route_v211] Querying both indexes for {len(tasks)} tasks ...", file=sys.stderr)

    for task in tasks:
        tid   = task["id"]
        fam   = task.get("family", "unknown")
        query = task["prompt"]

        c_score = get_top1_score(query, champion_index)
        r_score = get_top1_score(query, repair_index)
        margin  = r_score - c_score

        # Routing decisions
        f_dec = family_route(fam)
        c_dec = confidence_route(c_score, r_score, args.confidence_threshold, args.margin_threshold)

        # Oracle from v2.10 results
        c_pass = champion_v210.get(tid, None)
        r_pass = repair_v210.get(tid, None)
        if c_pass is not None and r_pass is not None:
            o_dec = oracle_route(c_pass, r_pass)
        else:
            o_dec = "unknown"

        # Record
        score_rows.append({
            "task_id":            tid,
            "family":             fam,
            "champion_top1":      f"{c_score:.4f}",
            "repair_top1":        f"{r_score:.4f}",
            "margin":             f"{margin:+.4f}",
            "family_route":       f_dec,
            "confidence_route":   c_dec,
            "oracle_route":       o_dec,
            "champion_v210_pass": c_pass,
            "repair_v210_pass":   r_pass,
        })
        decisions[tid] = {
            "family_router":     f_dec,
            "confidence_router": c_dec,
            "oracle_router":     o_dec,
        }

        # Split into sub-files
        if f_dec == "champion":
            fam_champion.append(task)
        else:
            fam_repair.append(task)

        if c_dec == "champion":
            conf_champion.append(task)
        else:
            conf_repair.append(task)

    # Write outputs
    write_routing_scores_csv(score_rows, output_dir / "routing_scores.csv")
    (output_dir / "routing_decisions.json").write_text(
        json.dumps(decisions, indent=2) + "\n"
    )
    print(f"[route_v211] Written routing decisions → {output_dir / 'routing_decisions.json'}",
          file=sys.stderr)

    write_jsonl(fam_champion,  output_dir / "tasks_family_champion.jsonl")
    write_jsonl(fam_repair,    output_dir / "tasks_family_repair.jsonl")
    write_jsonl(conf_champion, output_dir / "tasks_conf_champion.jsonl")
    write_jsonl(conf_repair,   output_dir / "tasks_conf_repair.jsonl")

    # Print routing summary
    n = len(tasks)
    print(f"\n[route_v211] Routing summary ({n} tasks):", file=sys.stderr)
    print(f"  Family router:     {len(fam_champion)} → champion, {len(fam_repair)} → repair",
          file=sys.stderr)
    print(f"  Confidence router: {len(conf_champion)} → champion, {len(conf_repair)} → repair",
          file=sys.stderr)
    oracle_repair = sum(1 for r in score_rows if r["oracle_route"] == "repair")
    print(f"  Oracle router:     {n - oracle_repair} → champion, {oracle_repair} → repair",
          file=sys.stderr)
    print(f"\n  Confidence thresholds: repair_top1 >= {args.confidence_threshold}"
          f", margin >= {args.margin_threshold}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
