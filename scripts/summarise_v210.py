#!/usr/bin/env python3
"""
scripts/summarise_v210.py
AetherForge v2.10 — Clean Repair-Generalisation Benchmark

Compares clean champion (memory/index_adapted) vs repair index (memory/index_adapted_v29)
on 32 untouched tasks across 5 pattern families.

Produces:
  results/v210_clean_repair_generalisation/summary.md
  results/v210_clean_repair_generalisation/per_family_breakdown.md
  results/v210_clean_repair_generalisation/claim_boundary.md
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_CHAMPION_SCORE = 23
TOTAL_TASKS_28       = 28
REPAIR_DIAGNOSTIC    = 27

FAMILIES = {
    "interval_merge":    "Interval merging and scheduling",
    "sorted_selection":  "Sorted-array / median / kth / merge",
    "nested_dict":       "Nested dictionary access and update",
    "tuple_tree":        "Tuple-tree recursion and structural traversal",
    "rle_encoding":      "Run-length encoding and structural string",
}

V29_BASELINE_CLEAN = {"score": 4, "total": 5, "pct": 80.0}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _pass_count(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("passed","").strip().lower() in ("true","1","yes","pass"))


def _family_breakdown(rows: list[dict], task_meta: dict[str, str]) -> dict[str, dict]:
    by_family: dict[str, dict] = {}
    for r in rows:
        tid = r.get("task_id", r.get("id", ""))
        fam = task_meta.get(tid, "unknown")
        passed = r.get("passed","").strip().lower() in ("true","1","yes","pass")
        if fam not in by_family:
            by_family[fam] = {"pass": 0, "total": 0, "tasks": []}
        by_family[fam]["total"] += 1
        if passed:
            by_family[fam]["pass"] += 1
        by_family[fam]["tasks"].append((tid, "PASS" if passed else "FAIL"))
    return by_family


def load_task_meta(tasks_file: Path) -> dict[str, str]:
    import json
    if not tasks_file.exists():
        return {}
    result = {}
    with open(tasks_file) as f:
        for line in f:
            t = json.loads(line)
            result[t["id"]] = t.get("family", "unknown")
    return result


def build_summary(
    champion_rows: list[dict],
    repair_rows: list[dict],
    task_meta: dict[str, str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    n_champ   = len(champion_rows)
    p_champ   = _pass_count(champion_rows)
    pct_champ = 100.0 * p_champ / n_champ if n_champ else 0.0

    n_repair   = len(repair_rows)
    p_repair   = _pass_count(repair_rows)
    pct_repair = 100.0 * p_repair / n_repair if n_repair else 0.0

    champ_by_fam  = _family_breakdown(champion_rows, task_meta)
    repair_by_fam = _family_breakdown(repair_rows, task_meta)

    # ── summary.md ──────────────────────────────────────────────────────────
    lines = [
        "# AetherForge v2.10 — Clean Repair-Generalisation Benchmark",
        f"\nGenerated: {now}\n",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Tasks | 32 tasks, 5 families, no overlap with frozen 28-task benchmark |",
        f"| Lane 1 | `memory/index_adapted` (99 records, clean champion) |",
        f"| Lane 2 | `memory/index_adapted_v29` (103 records, repair-enhanced) |",
        "",
        "## Score comparison",
        "",
        "| Lane | Score | vs v2.9 clean baseline (4/5=80%) |",
        "|------|-------|----------------------------------|",
    ]

    if champion_rows:
        delta = f"{pct_champ - V29_BASELINE_CLEAN['pct']:+.1f} pp"
        lines.append(f"| Champion index | {p_champ}/{n_champ} = {pct_champ:.1f}% | {delta} |")
    else:
        lines.append("| Champion index | _pending_ | — |")

    if repair_rows:
        delta = f"{pct_repair - V29_BASELINE_CLEAN['pct']:+.1f} pp"
        lines.append(f"| Repair index | {p_repair}/{n_repair} = {pct_repair:.1f}% | {delta} |")
    else:
        lines.append("| Repair index | _pending_ | — |")

    lines += [
        "",
        "Reference: clean champion 23/28 = 82.1%, repair diagnostic 27/28 = 96.4%,",
        "v2.9 clean generalisation baseline 4/5 = 80.0%.",
        "",
        "## Interpretation",
        "",
    ]

    if repair_rows and champion_rows:
        if pct_repair >= 75.0 and pct_repair > pct_champ:
            lines += [
                f"**Repair index ({pct_repair:.1f}%) beats champion index ({pct_champ:.1f}%) "
                f"by {pct_repair - pct_champ:.1f} pp on clean untouched tasks.**",
                "",
                "This is strong evidence that repair memory generalises beyond the frozen benchmark failures.",
                "The improvement is measured on 32 tasks that were never used in training, benchmark evaluation,",
                "or repair-index construction.",
            ]
        elif pct_repair >= 75.0:
            lines += [
                f"Both indexes score >= 75% ({pct_champ:.1f}% vs {pct_repair:.1f}%).",
                "The repair index does not clearly outperform the champion on clean tasks.",
                "Generalisation claim is weak; repair memory helps diagnostically but the champion",
                "index already retrieves useful context for these task families.",
            ]
        else:
            lines += [
                f"Repair index {pct_repair:.1f}% is below the 75% target on clean tasks.",
                "Repair memory does not generalise strongly beyond the known failure patterns.",
            ]
    else:
        lines.append("_Run `make eval-v210-clean-champion` and `make eval-v210-repair-index` to populate._")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[summarise_v210] Written: {output_dir / 'summary.md'}", file=sys.stderr)

    # ── per_family_breakdown.md ─────────────────────────────────────────────
    fam_lines = [
        "# v2.10 Per-Family Breakdown",
        f"\nGenerated: {now}\n",
    ]

    all_families = sorted(set(list(champ_by_fam.keys()) + list(repair_by_fam.keys())))
    fam_lines += [
        "| Family | Champion pass/total | Repair pass/total | Delta |",
        "|--------|---------------------|-------------------|-------|",
    ]
    for fam in sorted(FAMILIES.keys()):
        c = champ_by_fam.get(fam, {"pass": 0, "total": 0})
        r = repair_by_fam.get(fam, {"pass": 0, "total": 0})
        c_str = f"{c['pass']}/{c['total']}" if c["total"] else "—"
        r_str = f"{r['pass']}/{r['total']}" if r["total"] else "—"
        if c["total"] and r["total"]:
            delta_str = f"{100*(r['pass']/r['total'] - c['pass']/c['total']):.1f} pp"
        else:
            delta_str = "—"
        fam_lines.append(f"| {FAMILIES[fam]} | {c_str} | {r_str} | {delta_str} |")

    fam_lines.append("")
    for fam in sorted(FAMILIES.keys()):
        fam_lines.append(f"\n### {FAMILIES[fam]} (`{fam}`)\n")
        c = champ_by_fam.get(fam, {"tasks": []})
        r = repair_by_fam.get(fam, {"tasks": []})
        r_map = {tid: result for tid, result in r.get("tasks", [])}
        if c.get("tasks"):
            fam_lines.append("| Task | Champion | Repair |")
            fam_lines.append("|------|----------|--------|")
            for tid, c_result in c.get("tasks", []):
                r_result = r_map.get(tid, "—")
                fam_lines.append(f"| `{tid}` | {c_result} | {r_result} |")

    (output_dir / "per_family_breakdown.md").write_text("\n".join(fam_lines) + "\n")
    print(f"[summarise_v210] Written: {output_dir / 'per_family_breakdown.md'}", file=sys.stderr)

    # ── claim_boundary.md ───────────────────────────────────────────────────
    claim_lines = [
        "# AetherForge v2.10 — Claim Boundary",
        f"\nGenerated: {now}\n",
        "## Allowed claims",
        "",
        "1. **Clean champion baseline** (Lane 1): 23/28 = 82.1% on the frozen benchmark.",
        "   This is the authoritative performance ceiling for the model without repair.",
        "",
        "2. **Repair diagnostic** (Lane 2): 27/28 = 96.4% on the same frozen benchmark.",
        "   NOT a clean held-out champion. Repair records target known benchmark failures.",
        "",
        "3. **v2.10 clean generalisation** (both lanes on 32 untouched tasks):",
    ]
    if champion_rows and repair_rows:
        claim_lines += [
            f"   - Champion index: {p_champ}/{n_champ} = {pct_champ:.1f}%",
            f"   - Repair index:   {p_repair}/{n_repair} = {pct_repair:.1f}%",
        ]
        if pct_repair > pct_champ + 5.0 and pct_repair >= 75.0:
            claim_lines.append(
                "   - Claim supported: repair memory generalises to similar unseen tasks."
            )
        else:
            claim_lines.append(
                "   - Generalisation claim is limited; see per-family breakdown."
            )
    else:
        claim_lines.append("   - Results pending.")

    claim_lines += [
        "",
        "## Not allowed claims",
        "",
        "- Claiming 96.4% as a clean held-out benchmark score.",
        "- Claiming production-grade coding-agent reliability.",
        "- Comparing to SWE-bench, MBPP, or HumanEval without running those benchmarks.",
        "- AGI, quantum reasoning, or general superiority claims.",
        "",
        "## Promotion path to a new clean champion",
        "",
        "1. Merge `memory/raw_v29_repair/repair_records.jsonl` into `memory/raw_adapted/`.",
        "2. Rebuild `memory/index_adapted`.",
        "3. Run the full 28-task frozen benchmark with the updated index.",
        "4. Score >= 24/28 with no other changes = new clean champion.",
        "5. The 32-task v2.10 result remains a valid generalisation measure.",
    ]

    (output_dir / "claim_boundary.md").write_text("\n".join(claim_lines) + "\n")
    print(f"[summarise_v210] Written: {output_dir / 'claim_boundary.md'}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise v2.10 clean repair-generalisation results")
    p.add_argument("--champion-csv",
                   default="results/v210_clean_repair_generalisation/champion_results.csv")
    p.add_argument("--repair-csv",
                   default="results/v210_clean_repair_generalisation/repair_results.csv")
    p.add_argument("--tasks-file",
                   default="data/v210_clean_repair_generalisation_tasks.jsonl")
    p.add_argument("--output-dir",
                   default="results/v210_clean_repair_generalisation")
    args = p.parse_args()

    output_dir    = Path(args.output_dir)
    champion_rows = _read_csv(Path(args.champion_csv))
    repair_rows   = _read_csv(Path(args.repair_csv))
    task_meta     = load_task_meta(Path(args.tasks_file))

    build_summary(champion_rows, repair_rows, task_meta, output_dir)
    print(f"Done. See {output_dir}/summary.md")


if __name__ == "__main__":
    main()
