#!/usr/bin/env python3
"""
scripts/summarise_v211.py
AetherForge v2.11 — Retrieval Routing and Gating Audit

Merges per-router CSV results and produces:
  results/v211_retrieval_routing/summary.md
  results/v211_retrieval_routing/per_family_breakdown.md
  results/v211_retrieval_routing/per_task_routing.csv
  results/v211_retrieval_routing/claim_boundary.md
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FAMILIES = {
    "interval_merge":   "Interval merging and scheduling",
    "sorted_selection": "Sorted-array / median / kth / merge",
    "nested_dict":      "Nested dictionary access and update",
    "tuple_tree":       "Tuple-tree recursion and structural traversal",
    "rle_encoding":     "Run-length encoding and structural string",
}

# v2.10 baselines (hard-coded from completed run)
V210_CHAMPION = {"pass": 20, "total": 32, "pct": 62.5}
V210_REPAIR   = {"pass": 18, "total": 32, "pct": 56.2}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def is_passed(row: dict) -> bool:
    return row.get("passed", "").strip().lower() in ("true", "1", "yes", "pass")


def merge_evals(champion_rows: list[dict], repair_rows: list[dict]) -> dict[str, bool]:
    """Merge champion and repair sub-eval results into a single task_id → passed map."""
    result: dict[str, bool] = {}
    for r in champion_rows:
        tid = r.get("id", r.get("task_id", "")).strip()
        if tid:
            result[tid] = is_passed(r)
    for r in repair_rows:
        tid = r.get("id", r.get("task_id", "")).strip()
        if tid:
            result[tid] = is_passed(r)
    return result


def count_pass(passed_map: dict[str, bool]) -> int:
    return sum(1 for v in passed_map.values() if v)


def oracle_passed_map(routing_scores: list[dict]) -> dict[str, bool]:
    """Oracle: per task, use the index that passed; champion wins if both pass or both fail."""
    result = {}
    for row in routing_scores:
        tid        = row.get("task_id", "").strip()
        oracle_dec = row.get("oracle_route", "champion")
        c_pass_str = str(row.get("champion_v210_pass", "")).lower()
        r_pass_str = str(row.get("repair_v210_pass", "")).lower()
        c_pass = c_pass_str in ("true", "1", "yes", "pass")
        r_pass = r_pass_str in ("true", "1", "yes", "pass")
        if oracle_dec == "repair":
            result[tid] = r_pass
        else:
            result[tid] = c_pass
    return result


def load_task_meta(tasks_file: Path) -> dict[str, str]:
    if not tasks_file.exists():
        return {}
    result = {}
    with open(tasks_file) as f:
        for line in f:
            t = json.loads(line)
            result[t["id"]] = t.get("family", "unknown")
    return result


def family_breakdown(
    passed_map: dict[str, bool], task_meta: dict[str, str]
) -> dict[str, dict]:
    by_fam: dict[str, dict] = {}
    for tid, passed in passed_map.items():
        fam = task_meta.get(tid, "unknown")
        if fam not in by_fam:
            by_fam[fam] = {"pass": 0, "total": 0}
        by_fam[fam]["total"] += 1
        if passed:
            by_fam[fam]["pass"] += 1
    return by_fam


def build_outputs(
    fam_champion_rows:  list[dict],
    fam_repair_rows:    list[dict],
    conf_champion_rows: list[dict],
    conf_repair_rows:   list[dict],
    routing_scores:     list[dict],
    task_meta:          dict[str, str],
    output_dir:         Path,
    champion_index_size: int = 99,
    repair_index_size:   int = 103,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Compute per-router results
    fam_passed   = merge_evals(fam_champion_rows, fam_repair_rows)
    conf_passed  = merge_evals(conf_champion_rows, conf_repair_rows)
    oracle_passed = oracle_passed_map(routing_scores) if routing_scores else {}

    n = max(
        len(fam_passed) or 32,
        len(conf_passed) or 32,
        len(oracle_passed) or 32,
        32,
    )

    def score_str(passed_map: dict[str, bool]) -> str:
        if not passed_map:
            return "_pending_"
        p = count_pass(passed_map)
        t = max(len(passed_map), 1)
        return f"{p}/{t} = {100*p/t:.1f}%"

    def pct(passed_map: dict[str, bool]) -> float:
        if not passed_map:
            return 0.0
        p = count_pass(passed_map)
        return 100.0 * p / max(len(passed_map), 1)

    # ── summary.md ──────────────────────────────────────────────────────────
    lines = [
        "# AetherForge v2.11 — Retrieval Routing and Gating Audit",
        f"\nGenerated: {now}\n",
        "## v2.10 baselines (32 clean tasks)",
        "",
        "| Lane | Score | % |",
        "|---|---|---|",
        f"| Champion index ({champion_index_size} rec) | {V210_CHAMPION['pass']}/{V210_CHAMPION['total']} | {V210_CHAMPION['pct']:.1f}% |",
        f"| Repair index ({repair_index_size} rec)   | {V210_REPAIR['pass']}/{V210_REPAIR['total']} | {V210_REPAIR['pct']:.1f}% |",
        "",
        "## Routing results (same 32 clean tasks)",
        "",
        "| Router | Score | vs Champion | vs Repair |",
        "|--------|-------|-------------|-----------|",
    ]

    for label, passed_map in [
        ("Family router", fam_passed),
        ("Confidence router", conf_passed),
        ("Oracle router (ceiling)", oracle_passed),
    ]:
        s = score_str(passed_map)
        if passed_map:
            delta_c = f"{pct(passed_map) - V210_CHAMPION['pct']:+.1f} pp"
            delta_r = f"{pct(passed_map) - V210_REPAIR['pct']:+.1f} pp"
        else:
            delta_c = delta_r = "—"
        lines.append(f"| {label} | {s} | {delta_c} | {delta_r} |")

    lines += ["", "## Interpretation", ""]

    # Family router interpretation
    if fam_passed:
        fp = pct(fam_passed)
        lines.append(f"**Family router: {score_str(fam_passed)}**")
        if fp > V210_CHAMPION['pct'] + 5.0:
            lines += [
                f"Family routing beats champion by {fp - V210_CHAMPION['pct']:.1f} pp.",
                "Using repair index for interval_merge tasks improves overall performance.",
            ]
        elif fp > V210_CHAMPION['pct']:
            lines += [
                f"Family routing improves by {fp - V210_CHAMPION['pct']:.1f} pp over champion.",
                "Marginal gain — routing to repair index for interval_merge helps slightly.",
            ]
        else:
            lines += [
                "Family routing does not improve over champion index.",
                "The routing gain in interval_merge is offset by other variance.",
            ]
        lines.append("")

    # Confidence router interpretation
    if conf_passed:
        cp = pct(conf_passed)
        lines.append(f"**Confidence router: {score_str(conf_passed)}**")
        if cp > V210_CHAMPION['pct'] + 5.0:
            lines += [
                f"Confidence routing beats champion by {cp - V210_CHAMPION['pct']:.1f} pp.",
                "Selective routing based on retrieval score margin is effective.",
            ]
        elif cp >= V210_CHAMPION['pct']:
            lines += [
                f"Confidence routing matches or slightly beats champion ({cp:.1f}% vs {V210_CHAMPION['pct']:.1f}%).",
                "Conservative threshold preserves champion performance; marginal improvement.",
            ]
        else:
            lines += [
                f"Confidence routing ({cp:.1f}%) falls below champion ({V210_CHAMPION['pct']:.1f}%).",
                "The confidence threshold routes too many tasks to repair, causing regressions.",
            ]
        lines.append("")

    # Oracle interpretation
    if oracle_passed:
        op = pct(oracle_passed)
        lines.append(f"**Oracle router (ceiling): {score_str(oracle_passed)}**")
        gap = op - V210_CHAMPION['pct']
        lines += [
            f"Oracle router beats champion by {gap:.1f} pp.",
            f"This is the theoretical ceiling for any index-selection routing scheme.",
        ]
        if gap > 15.0:
            lines.append(
                "Large oracle gap suggests index selection is a significant bottleneck."
            )
        elif gap > 5.0:
            lines.append(
                "Moderate oracle gap — routing can help but the ceiling is limited."
            )
        else:
            lines.append(
                "Small oracle gap — index selection provides minimal additional headroom."
            )
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[summarise_v211] Written: {output_dir / 'summary.md'}", file=sys.stderr)

    # ── per_family_breakdown.md ─────────────────────────────────────────────
    fam_lines = [
        "# v2.11 Per-Family Breakdown",
        f"\nGenerated: {now}\n",
        "| Family | Champion | Repair | Family-router | Confidence-router | Oracle |",
        "|--------|----------|--------|---------------|-------------------|--------|",
    ]

    # Hardcoded v2.10 per-family results
    v210_champ_by_fam = {
        "interval_merge":   {"pass": 4, "total": 6},
        "sorted_selection": {"pass": 3, "total": 7},
        "nested_dict":      {"pass": 6, "total": 7},
        "tuple_tree":       {"pass": 3, "total": 7},
        "rle_encoding":     {"pass": 4, "total": 5},
    }
    v210_repair_by_fam = {
        "interval_merge":   {"pass": 5, "total": 6},
        "sorted_selection": {"pass": 3, "total": 7},
        "nested_dict":      {"pass": 5, "total": 7},
        "tuple_tree":       {"pass": 2, "total": 7},
        "rle_encoding":     {"pass": 3, "total": 5},
    }

    fam_fam    = family_breakdown(fam_passed, task_meta)
    conf_fam   = family_breakdown(conf_passed, task_meta)
    oracle_fam = family_breakdown(oracle_passed, task_meta)

    def fmt_fam(d: dict, fam: str) -> str:
        if fam not in d or not d[fam]["total"]:
            return "—"
        v = d[fam]
        return f"{v['pass']}/{v['total']}"

    for fam, fam_name in sorted(FAMILIES.items()):
        c = v210_champ_by_fam.get(fam, {"pass": 0, "total": 0})
        r = v210_repair_by_fam.get(fam, {"pass": 0, "total": 0})
        fam_lines.append(
            f"| {fam_name} "
            f"| {c['pass']}/{c['total']} "
            f"| {r['pass']}/{r['total']} "
            f"| {fmt_fam(fam_fam, fam)} "
            f"| {fmt_fam(conf_fam, fam)} "
            f"| {fmt_fam(oracle_fam, fam)} |"
        )

    (output_dir / "per_family_breakdown.md").write_text("\n".join(fam_lines) + "\n")
    print(f"[summarise_v211] Written: {output_dir / 'per_family_breakdown.md'}", file=sys.stderr)

    # ── per_task_routing.csv ────────────────────────────────────────────────
    if routing_scores:
        fields = [
            "task_id", "family",
            "champion_top1", "repair_top1", "margin",
            "family_route", "confidence_route", "oracle_route",
            "champion_v210_pass", "repair_v210_pass",
            "family_router_pass", "conf_router_pass", "oracle_router_pass",
        ]
        routing_csv_rows = []
        for row in routing_scores:
            tid = row.get("task_id", "")
            routing_csv_rows.append({
                "task_id":             tid,
                "family":              row.get("family", ""),
                "champion_top1":       row.get("champion_top1", ""),
                "repair_top1":         row.get("repair_top1", ""),
                "margin":              row.get("margin", ""),
                "family_route":        row.get("family_route", ""),
                "confidence_route":    row.get("confidence_route", ""),
                "oracle_route":        row.get("oracle_route", ""),
                "champion_v210_pass":  row.get("champion_v210_pass", ""),
                "repair_v210_pass":    row.get("repair_v210_pass", ""),
                "family_router_pass":  fam_passed.get(tid, ""),
                "conf_router_pass":    conf_passed.get(tid, ""),
                "oracle_router_pass":  oracle_passed.get(tid, ""),
            })
        with open(output_dir / "per_task_routing.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(routing_csv_rows)
        print(f"[summarise_v211] Written: {output_dir / 'per_task_routing.csv'}", file=sys.stderr)

    # ── claim_boundary.md ───────────────────────────────────────────────────
    claim_lines = [
        "# AetherForge v2.11 — Claim Boundary",
        f"\nGenerated: {now}\n",
        "## Experiment history",
        "",
        "| Version | Result | Label |",
        "|---------|--------|-------|",
        "| v2.9 repair diagnostic | 27/28 = 96.4% | Diagnostic on frozen benchmark |",
        "| v2.10 champion (32 clean) | 20/32 = 62.5% | Clean generalisation |",
        "| v2.10 repair (32 clean) | 18/32 = 56.2% | Rejected: global repair regresses |",
        "| v2.11 family router | " + score_str(fam_passed) + " | Routing experiment |",
        "| v2.11 confidence router | " + score_str(conf_passed) + " | Routing experiment |",
        "| v2.11 oracle ceiling | " + score_str(oracle_passed) + " | Diagnostic ceiling |",
        "",
        "## Allowed claims",
        "",
        "1. **Clean champion**: 23/28 = 82.1% on frozen 28-task benchmark (unchanged).",
        "2. **v2.10 clean generalisation baseline**: 20/32 = 62.5% (champion index).",
        "3. **Retrieval routing ceiling** (oracle): up to " + score_str(oracle_passed) + " with perfect per-task routing.",
    ]
    if fam_passed:
        claim_lines.append(
            f"4. **Family router**: {score_str(fam_passed)} — conservative routing using task family only."
        )
    if conf_passed:
        claim_lines.append(
            f"5. **Confidence router**: {score_str(conf_passed)} — routing based on retrieval score margin."
        )

    claim_lines += [
        "",
        "## Not allowed claims",
        "",
        "- Claiming the 96.4% diagnostic as a clean held-out score.",
        "- Claiming global repair-index promotion improves clean generalisation (v2.10 ruled this out).",
        "- Claiming a new frozen 28-task champion (champion index is unchanged at 23/28).",
        "- Claiming any routing strategy makes the model 'production-ready'.",
        "",
        "## Next steps (if routing is effective)",
        "",
        "- If family or confidence router beats champion by >= 5 pp:",
        "  Implement routing as an inference-time feature; test on the frozen 28-task benchmark.",
        "- If oracle router is >> 70% but deployed routers are << 65%:",
        "  The routing signal is too noisy; need better task-family classification.",
        "- If all routers ~ champion: retrieval noise is not the binding constraint.",
        "  Consider model improvement (more training data, better LoRA recipe) instead.",
    ]

    (output_dir / "claim_boundary.md").write_text("\n".join(claim_lines) + "\n")
    print(f"[summarise_v211] Written: {output_dir / 'claim_boundary.md'}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise v2.11 retrieval routing results")
    p.add_argument("--routing-scores-csv",
                   default="results/v211_retrieval_routing/routing_scores.csv")
    p.add_argument("--fam-champion-csv",
                   default="outputs/eval_v211_family_champion/best_of_3.csv")
    p.add_argument("--fam-repair-csv",
                   default="outputs/eval_v211_family_repair/best_of_3.csv")
    p.add_argument("--conf-champion-csv",
                   default="outputs/eval_v211_conf_champion/best_of_3.csv")
    p.add_argument("--conf-repair-csv",
                   default="outputs/eval_v211_conf_repair/best_of_3.csv")
    p.add_argument("--tasks-file",
                   default="data/v210_clean_repair_generalisation_tasks.jsonl")
    p.add_argument("--output-dir",
                   default="results/v211_retrieval_routing")
    args = p.parse_args()

    routing_scores     = read_csv(Path(args.routing_scores_csv))
    fam_champion_rows  = read_csv(Path(args.fam_champion_csv))
    fam_repair_rows    = read_csv(Path(args.fam_repair_csv))
    conf_champion_rows = read_csv(Path(args.conf_champion_csv))
    conf_repair_rows   = read_csv(Path(args.conf_repair_csv))
    task_meta          = load_task_meta(Path(args.tasks_file))

    build_outputs(
        fam_champion_rows, fam_repair_rows,
        conf_champion_rows, conf_repair_rows,
        routing_scores, task_meta,
        Path(args.output_dir),
    )
    print(f"Done. See {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
