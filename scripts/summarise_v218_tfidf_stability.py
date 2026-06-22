"""
scripts/summarise_v218_tfidf_stability.py

Read three TF-IDF 32-task eval CSVs, compute per-task stability,
and write results/v218_retrieval_stability/tfidf_stability.csv.

Usage:
    python scripts/summarise_v218_tfidf_stability.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RUN_DIRS = [
    Path("outputs/eval_v218_tfidf_32_run1"),
    Path("outputs/eval_v218_tfidf_32_run2"),
    Path("outputs/eval_v218_tfidf_32_run3"),
]
OUT_DIR = Path("results/v218_retrieval_stability")
HISTORICAL = {"n_pass": 20, "n_tasks": 32, "label": "v2.10 historical"}


def load_csv(run_dir: Path) -> dict[str, bool]:
    """Return {task_id: passed} from a best_of_3.csv in run_dir."""
    csv_path = run_dir / "best_of_3.csv"
    if not csv_path.exists():
        return {}
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id") or row.get("id") or row.get("task", "")
            passed_raw = row.get("passed", "False")
            passed = passed_raw.strip().lower() in ("true", "1", "yes")
            if task_id:
                result[task_id] = passed
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, bool]] = []
    run_labels = []
    for i, d in enumerate(RUN_DIRS, 1):
        data = load_csv(d)
        if not data:
            print(f"  WARNING: no data found in {d} — run target first")
        else:
            print(f"  run{i}: {d.name}  n={len(data)}  pass={sum(data.values())}")
        runs.append(data)
        run_labels.append(f"run{i}")

    # Union of all task ids
    all_tasks = sorted(set().union(*[r.keys() for r in runs]))
    if not all_tasks:
        print("No task data found. Run eval targets first.")
        sys.exit(1)

    # Per-task stability
    rows = []
    for task_id in all_tasks:
        passes = [int(r.get(task_id, False)) for r in runs]
        n_pass = sum(passes)
        n_runs = len([r for r in runs if task_id in r])
        if n_runs == 0:
            stability = "missing"
        elif n_pass == n_runs:
            stability = "stable_pass"
        elif n_pass == 0:
            stability = "stable_fail"
        else:
            stability = "flip"
        rows.append({
            "task_id": task_id,
            **{f"run{i+1}": passes[i] if i < len(passes) else "" for i in range(3)},
            "n_pass": n_pass,
            "n_runs": n_runs,
            "pass_rate": f"{n_pass}/{n_runs}",
            "stability": stability,
        })

    # Write stability CSV
    stability_path = OUT_DIR / "tfidf_stability.csv"
    with open(stability_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "run1", "run2", "run3", "n_pass", "n_runs", "pass_rate", "stability"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {stability_path}")

    # Summary stats
    n_stable_pass = sum(1 for r in rows if r["stability"] == "stable_pass")
    n_stable_fail = sum(1 for r in rows if r["stability"] == "stable_fail")
    n_flip = sum(1 for r in rows if r["stability"] == "flip")
    flip_tasks = [r["task_id"] for r in rows if r["stability"] == "flip"]

    # Per-run totals
    run_totals = []
    for i, run in enumerate(runs):
        total = sum(run.values())
        run_totals.append(total)
        print(f"  run{i+1}: {total}/{len(run)} = {100*total/max(len(run),1):.1f}%")

    valid_totals = [t for t in run_totals if t > 0]
    if valid_totals:
        mean_pass = sum(valid_totals) / len(valid_totals)
        var_pass = max(valid_totals) - min(valid_totals)
        print(f"\n  mean: {mean_pass:.1f}/{len(all_tasks)}  range: ±{var_pass/2:.1f}  ({min(valid_totals)}–{max(valid_totals)})")
    print(f"  stable_pass={n_stable_pass}  stable_fail={n_stable_fail}  flip={n_flip}")
    print(f"  flip tasks: {flip_tasks}")
    print(f"\n  Historical champion (v2.10): {HISTORICAL['n_pass']}/{HISTORICAL['n_tasks']} = {100*HISTORICAL['n_pass']/HISTORICAL['n_tasks']:.1f}%")

    # Write flip CSV
    flip_path = OUT_DIR / "per_task_flips.csv"
    with open(flip_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "run1", "run2", "run3", "pass_rate", "stability"])
        w.writeheader()
        for r in rows:
            if r["stability"] in ("flip",):
                w.writerow({k: r[k] for k in ["task_id", "run1", "run2", "run3", "pass_rate", "stability"]})
    print(f"Wrote {flip_path}")

    # Update summary.md
    _write_summary(rows, run_totals, flip_tasks, n_stable_pass, n_stable_fail, n_flip, all_tasks)


def _write_summary(rows, run_totals, flip_tasks, n_stable_pass, n_stable_fail, n_flip, all_tasks):
    valid_totals = [t for t in run_totals if t > 0]
    mean_pass = sum(valid_totals) / len(valid_totals) if valid_totals else 0
    rng = (min(valid_totals), max(valid_totals)) if valid_totals else (0, 0)

    lines = [
        "# v2.18 Retrieval Stability — TF-IDF Baseline Summary",
        "",
        "## Three-run TF-IDF baseline (32-task clean benchmark)",
        "",
        f"| Run | n_pass | score |",
        "|---|---|---|",
    ]
    for i, t in enumerate(run_totals):
        n = len(all_tasks)
        lines.append(f"| TF-IDF run {i+1} | {t}/{n} | {100*t/max(n,1):.1f}% |")
    lines += [
        f"| **Historical champion (v2.10)** | **20/32** | **62.5%** |",
        "",
        f"**Mean across {len([t for t in run_totals if t>0])} runs:** {mean_pass:.1f}/32 = {100*mean_pass/32:.1f}%  ",
        f"**Range:** {rng[0]}–{rng[1]} (±{(rng[1]-rng[0])/2:.1f} tasks)  ",
        "",
        "## Task stability",
        "",
        f"- Stable PASS (all runs): {n_stable_pass}",
        f"- Stable FAIL (all runs): {n_stable_fail}",
        f"- Flip (inconsistent across runs): {n_flip}",
        "",
        "### Flip tasks (unstable — within sampling variance)",
        "",
    ]
    for t in flip_tasks:
        r = next(x for x in rows if x["task_id"] == t)
        lines.append(f"- `{t}`: passes {r['pass_rate']} runs")

    lines += [
        "",
        "## Implication for dense retrieval evaluation",
        "",
        "The noise floor is the range above. A dense retrieval improvement must exceed",
        "the max of (a) this range and (b) 2 tasks to be considered real.",
        "",
        "See `per_task_flips.csv` for full flip detail.",
        "See `tfidf_stability.csv` for per-task pass rates.",
    ]

    summary_path = Path("results/v218_retrieval_stability/tfidf_baseline_report.md")
    summary_path.write_text("\n".join(lines))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
