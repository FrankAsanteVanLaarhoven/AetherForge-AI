#!/usr/bin/env python3
"""
scripts/summarise_v27_preservation.py
v2.7 Champion Preservation Audit

Compare the champion adapter result across eval configurations and against v2.6
baselines.  Produces three output files:
  - results/v27_champion_preservation/summary.md
  - results/v27_champion_preservation/per_task_comparison.csv
  - results/v27_champion_preservation/failure_diff.md

Usage:
    python scripts/summarise_v27_preservation.py \\
        --output-dir results/v27_champion_preservation
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict] | None:
    """Return rows as list of dicts, or None if the file does not exist."""
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


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

_THRESHOLDS = {
    "champion": 75.0,
    "merge_drop": 5.0,
    "memory_lift": 10.0,
    "env_delta": 3.0,
}


def _diagnose(results: dict[str, list[dict] | None]) -> list[str]:
    """Produce a list of diagnostic findings based on available results."""
    findings = []

    champ = results.get("champion")
    merged = results.get("merged")
    no_mem = results.get("no_memory")
    orig_mem = results.get("original_memory")
    mltorch = results.get("mltorch")
    aetherforge = results.get("aetherforge")

    if champ is not None:
        p, n, rate = _pass_rate(champ)
        findings.append(
            f"Champion adapter (unmerged, with memory): {p}/{n} = {rate:.1f}%"
        )
        if rate >= _THRESHOLDS["champion"]:
            findings.append("  CONFIRMED: champion reproduces >= 75.0%")
        else:
            findings.append(
                f"  WARNING: champion dropped to {rate:.1f}% "
                "(expected 75.0%) — check environment or data"
            )

    if merged is not None:
        mp, mn, mrate = _pass_rate(merged)
        findings.append(
            f"Merged champion (merge_and_unload, no retraining): {mp}/{mn} = {mrate:.1f}%"
        )
        if champ is not None:
            cp, cn, crate = _pass_rate(champ)
            delta = mrate - crate
            if abs(delta) <= _THRESHOLDS["merge_drop"]:
                findings.append(
                    f"  merge_and_unload is SAFE ({delta:+.1f} pp vs unmerged adapter)"
                )
            else:
                findings.append(
                    f"  merge_and_unload DAMAGED the model ({delta:+.1f} pp vs unmerged)"
                )
                findings.append(
                    "  Root cause candidate: merge precision / config / tokenizer drift"
                )

    if no_mem is not None and champ is not None:
        nm_p, nm_n, nm_rate = _pass_rate(no_mem)
        cp, cn, crate = _pass_rate(champ)
        lift = crate - nm_rate
        findings.append(
            f"Champion without memory: {nm_p}/{nm_n} = {nm_rate:.1f}%  "
            f"(memory lift = {lift:+.1f} pp)"
        )
        if lift >= _THRESHOLDS["memory_lift"]:
            findings.append(
                "  IMPORTANT: result is memory-dependent. "
                "The 75.0% claim reflects a model+memory system, not adapter alone."
            )
        else:
            findings.append(
                f"  Memory contributes only {lift:+.1f} pp — result is not purely memory-driven."
            )

    if mltorch is not None and aetherforge is not None:
        ml_p, ml_n, ml_rate = _pass_rate(mltorch)
        af_p, af_n, af_rate = _pass_rate(aetherforge)
        delta = af_rate - ml_rate
        findings.append(
            f"ml-torch env: {ml_p}/{ml_n} = {ml_rate:.1f}%"
        )
        findings.append(
            f"aetherforge-train env: {af_p}/{af_n} = {af_rate:.1f}%"
        )
        if abs(delta) > _THRESHOLDS["env_delta"]:
            findings.append(
                f"  ENV SENSITIVE: {delta:+.1f} pp gap between environments. "
                "Check CUDA version, torch version, dtype defaults."
            )
        else:
            findings.append(
                f"  Environment-stable within {abs(delta):.1f} pp."
            )

    if not findings:
        findings.append(
            "No eval results found yet. "
            "Run eval-v27-champion-adapter and other make targets first."
        )

    return findings


# ---------------------------------------------------------------------------
# Per-task comparison
# ---------------------------------------------------------------------------

def _build_task_table(results: dict[str, list[dict] | None]) -> list[dict]:
    """Build one row per task with pass/fail across all run configs."""
    task_ids: list[str] = []
    task_cats: dict[str, str] = {}
    seen: set[str] = set()
    for rows in results.values():
        if rows is None:
            continue
        for r in rows:
            tid = r.get("id", "")
            if tid and tid not in seen:
                task_ids.append(tid)
                task_cats[tid] = r.get("category", "")
                seen.add(tid)

    table = []
    for tid in task_ids:
        row: dict = {"id": tid, "category": task_cats.get(tid, "")}
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


def _quadrant(champion_pass: bool | None, other_pass: bool | None) -> str:
    if champion_pass is None or other_pass is None:
        return "unknown"
    if champion_pass and not other_pass:
        return "champion_only"
    if not champion_pass and other_pass:
        return "other_only"
    if champion_pass and other_pass:
        return "both_pass"
    return "both_fail"


# ---------------------------------------------------------------------------
# Failure diff
# ---------------------------------------------------------------------------

def _failure_diff(
    champion: list[dict] | None,
    others: dict[str, list[dict] | None],
) -> list[str]:
    """Produce per-task failure analysis comparing champion vs each other run."""
    lines = []
    if champion is None:
        lines.append("Champion results not yet available.")
        return lines

    champ_pass = {r["id"]: _passed(r) for r in champion}

    for label, rows in others.items():
        if rows is None:
            lines.append(f"\n### {label}: not yet evaluated")
            continue
        other_pass = {r["id"]: _passed(r) for r in rows}
        all_ids = sorted(set(champ_pass) | set(other_pass))

        champion_only = [t for t in all_ids
                         if champ_pass.get(t) and not other_pass.get(t)]
        other_only = [t for t in all_ids
                      if not champ_pass.get(t) and other_pass.get(t)]
        both_pass = [t for t in all_ids
                     if champ_pass.get(t) and other_pass.get(t)]
        both_fail = [t for t in all_ids
                     if not champ_pass.get(t) and not other_pass.get(t)]

        cp, cn, crate = _pass_rate(champion)
        op, on, orate = _pass_rate(rows)
        lines.append(f"\n### champion vs {label}")
        lines.append(f"champion: {cp}/{cn} = {crate:.1f}%")
        lines.append(f"{label}: {op}/{on} = {orate:.1f}%")
        lines.append(f"delta: {orate - crate:+.1f} pp")
        lines.append("")
        if champion_only:
            lines.append(f"champion PASS / {label} FAIL ({len(champion_only)} tasks):")
            for t in champion_only:
                lines.append(f"  {t}")
        if other_only:
            lines.append(f"champion FAIL / {label} PASS ({len(other_only)} tasks):")
            for t in other_only:
                lines.append(f"  {t}")
        if both_pass:
            lines.append(f"both PASS: {both_pass}")
        if both_fail:
            lines.append(f"both FAIL: {both_fail}")

    return lines


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_summary(findings: list[str], results: dict, output_dir: Path) -> Path:
    path = output_dir / "summary.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# AetherForge v2.7 — Champion Preservation Audit",
        "",
        f"Generated: {ts}",
        "",
        "## Research question",
        "",
        "Why does the original 300-step LoRA reach 75.0%, while merge-and-retrain",
        "runs fall to 57.1% or worse?",
        "",
        "## Known results",
        "",
        "| Configuration | Result |",
        "|---|---|",
        "| Champion 300-step LoRA + memory | 21/28 = 75.0% |",
        "| v2.6 traces000 | 16/28 = 57.1% |",
        "| v2.6 traces010 | 14/28 = 50.0% |",
        "| v2.6 traces025 | 15/28 = 53.6% |",
        "| v2.5 full trace blend | 15/28 = 53.6% |",
        "",
        "## v2.7 Audit results",
        "",
    ]

    labels = {
        "champion":        "Champion adapter (unmerged, best-of-3, memory/index)",
        "mltorch":         "Champion adapter in ml-torch env",
        "aetherforge":     "Champion adapter in aetherforge-train env",
        "merged":          "Merged champion (merge_and_unload, no retraining)",
        "no_memory":       "Champion adapter — memory DISABLED",
        "original_memory": "Champion adapter — memory/index (explicit)",
    }
    for key, desc in labels.items():
        rows = results.get(key)
        if rows is not None:
            p, n, rate = _pass_rate(rows)
            lines.append(f"- **{desc}**: {p}/{n} = {rate:.1f}%")
        else:
            lines.append(f"- {desc}: *not yet evaluated*")

    lines += [
        "",
        "## Diagnostic findings",
        "",
    ]
    for f in findings:
        lines.append(f"- {f}")

    lines += [
        "",
        "## Preservation hypothesis matrix",
        "",
        "| Hypothesis | Evidence |",
        "|---|---|",
    ]
    champ = results.get("champion")
    merged = results.get("merged")
    no_mem = results.get("no_memory")

    if champ is not None and merged is not None:
        _, _, cr = _pass_rate(champ)
        _, _, mr = _pass_rate(merged)
        d = mr - cr
        evidence = f"merge delta = {d:+.1f} pp"
        verdict = "likely safe" if abs(d) <= 5 else "DAMAGED"
        lines.append(f"| merge_and_unload damages model | {evidence} → {verdict} |")
    else:
        lines.append("| merge_and_unload damages model | pending — run eval-v27-merged-champion |")

    if champ is not None and no_mem is not None:
        _, _, cr = _pass_rate(champ)
        _, _, nr = _pass_rate(no_mem)
        lift = cr - nr
        verdict = "memory-dependent" if lift >= 10 else "not purely memory-driven"
        lines.append(f"| Result depends on memory | memory lift = {lift:+.1f} pp → {verdict} |")
    else:
        lines.append("| Result depends on memory | pending — run eval-v27-champion-no-memory |")

    mltorch = results.get("mltorch")
    af = results.get("aetherforge")
    if mltorch is not None and af is not None:
        _, _, ml = _pass_rate(mltorch)
        _, _, afr = _pass_rate(af)
        d = afr - ml
        verdict = "env-sensitive" if abs(d) > 3 else "env-stable"
        lines.append(f"| Environment-sensitive | env delta = {d:+.1f} pp → {verdict} |")
    else:
        lines.append("| Environment-sensitive | pending — run both env targets |")

    lines += [
        "",
        "## Claim boundary",
        "",
        "- Do NOT claim a new model improvement from this audit.",
        "- Do NOT claim SWE-bench success.",
        "- This audit only diagnoses why the champion is not preserved by merge-and-retrain.",
        "- Champion path: `outputs/qwen15b_memory_300steps/final`",
        "- Memory index: `memory/index`",
        "",
    ]

    path.write_text("\n".join(lines))
    return path


def _write_per_task_csv(table: list[dict], output_dir: Path) -> Path:
    path = output_dir / "per_task_comparison.csv"
    if not table:
        path.write_text("id,category\n")
        return path
    fields = list(table[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    return path


def _write_failure_diff(diff_lines: list[str], output_dir: Path) -> Path:
    path = output_dir / "failure_diff.md"
    header = [
        "# v2.7 Failure Diff — Champion vs Other Configurations",
        "",
        "Tasks where champion passes but the comparison fails reveal what",
        "behaviour was lost during merge-and-retrain.",
        "",
    ]
    path.write_text("\n".join(header + diff_lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="v2.7 Champion Preservation Audit summary"
    )
    ap.add_argument(
        "--champion-csv",
        default="outputs/eval_v27_champion_adapter/best_of_3.csv",
    )
    ap.add_argument(
        "--mltorch-csv",
        default="outputs/eval_v27_champion_adapter_mltorch/best_of_3.csv",
    )
    ap.add_argument(
        "--aetherforge-csv",
        default="outputs/eval_v27_champion_adapter_aetherforge/best_of_3.csv",
    )
    ap.add_argument(
        "--merged-csv",
        default="outputs/eval_v27_merged_champion/best_of_3.csv",
    )
    ap.add_argument(
        "--no-memory-csv",
        default="outputs/eval_v27_champion_no_memory/best_of_3.csv",
    )
    ap.add_argument(
        "--original-memory-csv",
        default="outputs/eval_v27_champion_original_memory/best_of_3.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="results/v27_champion_preservation",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict] | None] = {
        "champion":        _load_csv(Path(args.champion_csv)),
        "mltorch":         _load_csv(Path(args.mltorch_csv)),
        "aetherforge":     _load_csv(Path(args.aetherforge_csv)),
        "merged":          _load_csv(Path(args.merged_csv)),
        "no_memory":       _load_csv(Path(args.no_memory_csv)),
        "original_memory": _load_csv(Path(args.original_memory_csv)),
    }

    available = [k for k, v in results.items() if v is not None]
    missing   = [k for k, v in results.items() if v is None]
    print(f"\nv2.7 Preservation Audit — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Available: {available}")
    print(f"Missing  : {missing}")
    print()

    for key, rows in results.items():
        if rows is not None:
            p, n, rate = _pass_rate(rows)
            cats = _by_category(rows)
            print(f"  {key:20s}: {p}/{n} = {rate:.1f}%")
            for cat, (cp, cn) in sorted(cats.items()):
                print(f"    {cat:12s}: {cp}/{cn} = {100*cp//cn if cn else 0}%")

    print()
    findings = _diagnose(results)
    for f in findings:
        print(f"  {f}")

    # Build per-task table (all configs together)
    task_table = _build_task_table(results)

    # Failure diff: champion vs each other config (except mltorch/aetherforge vs each other)
    others = {
        k: results[k]
        for k in ("merged", "no_memory", "mltorch", "aetherforge")
    }
    diff_lines = _failure_diff(results.get("champion"), others)

    # Write outputs
    p1 = _write_summary(findings, results, out)
    p2 = _write_per_task_csv(task_table, out)
    p3 = _write_failure_diff(diff_lines, out)

    print(f"\nOutputs written:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")


if __name__ == "__main__":
    main()
