"""
scripts/summarise_v218_phaseb.py

Summarise v2.18 Phase B — code-specialised dense retrieval audit.

Reads three dense + three hybrid 32-task eval CSVs and the Phase A baseline
stability CSV, computes per-mode means/ranges and per-task / per-family deltas,
applies the strict promotion gate, and writes:

  results/v218_retrieval_stability/phase_b_code_dense_summary.md
  results/v218_retrieval_stability/phase_b_code_dense_comparison.csv
  results/v218_retrieval_stability/phase_b_per_task_matrix.csv
  results/v218_retrieval_stability/phase_b_per_family_breakdown.md
  results/v218_retrieval_stability/phase_b_claim_boundary.md

The Phase B encoder (UniXcoder/CodeBERT, 768d) is compared against the protected
code-aware MiniLM dense baseline (384d, mean 16.3/32). In hybrid mode the stage-1
shortlist uses memory/index_adapted — code-aware MiniLM dense, NOT TF-IDF.

Usage:
    python scripts/summarise_v218_phaseb.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path("results/v218_retrieval_stability")
BASELINE_CSV = OUT_DIR / "tfidf_stability.csv"  # Phase A per-task stability

MODE_RUN_DIRS = {
    "dense": [
        Path("outputs/eval_v218_phaseb_code_dense_32_run1"),
        Path("outputs/eval_v218_phaseb_code_dense_32_run2"),
        Path("outputs/eval_v218_phaseb_code_dense_32_run3"),
    ],
    "hybrid": [
        Path("outputs/eval_v218_phaseb_code_hybrid_32_run1"),
        Path("outputs/eval_v218_phaseb_code_hybrid_32_run2"),
        Path("outputs/eval_v218_phaseb_code_hybrid_32_run3"),
    ],
}

NOISE_FLOOR = 2.0   # tasks above baseline mean required for the promotion gate
TIE_BAND = 1.0      # |delta| within this band counts as a tie (flip-task noise)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_eval_csv(run_dir: Path) -> dict[str, bool]:
    """Return {task_id: passed} from best_of_3.csv in run_dir (empty if absent)."""
    csv_path = run_dir / "best_of_3.csv"
    if not csv_path.exists():
        return {}
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            task_id = row.get("task_id") or row.get("id") or row.get("task", "")
            passed = (row.get("passed", "False") or "").strip().lower() in ("true", "1", "yes")
            if task_id:
                result[task_id] = passed
    return result


def load_baseline() -> tuple[dict[str, int], dict[str, str], list[int]]:
    """Return (per-task n_pass, per-task stability, [run1,run2,run3 totals]) from Phase A."""
    if not BASELINE_CSV.exists():
        print(f"ERROR: baseline {BASELINE_CSV} not found — run summarise-v218-tfidf-stability first.")
        sys.exit(1)
    n_pass, stability = {}, {}
    totals = [0, 0, 0]
    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            tid = row["task_id"]
            n_pass[tid] = int(row["n_pass"])
            stability[tid] = row["stability"]
            for i in range(3):
                totals[i] += int(row.get(f"run{i+1}", 0) or 0)
    return n_pass, stability, totals


def family_of(task_id: str) -> str:
    """Family = first token after the v210_ prefix (e.g. tree_sum -> tree)."""
    stem = task_id.split("_", 1)[1] if task_id.startswith("v210_") else task_id
    return stem.split("_", 1)[0]


def stability_label(n_pass: int, n_runs: int) -> str:
    if n_runs == 0:
        return "missing"
    if n_pass == n_runs:
        return "stable_pass"
    if n_pass == 0:
        return "stable_fail"
    return "flip"


# ── per-mode aggregation ────────────────────────────────────────────────────────

def aggregate_mode(run_dirs: list[Path]) -> dict:
    """Return per-mode stats: per-task n_pass, run totals, mean, range, n_runs_present."""
    runs = [load_eval_csv(d) for d in run_dirs]
    present = [r for r in runs if r]
    all_tasks = sorted(set().union(*[set(r) for r in present])) if present else []
    per_task = {t: sum(int(r.get(t, False)) for r in present) for t in all_tasks}
    totals = [sum(r.values()) for r in present]
    mean = sum(totals) / len(totals) if totals else 0.0
    rng = (min(totals), max(totals)) if totals else (0, 0)
    return {
        "per_task": per_task,
        "n_runs": len(present),
        "totals": totals,
        "mean": mean,
        "range": rng,
        "tasks": all_tasks,
    }


def verdict_for(mode_mean: float, baseline_mean: float, gate: float) -> tuple[str, str]:
    """Return (short_tag, sentence) using the user's claim templates."""
    delta = mode_mean - baseline_mean
    if mode_mean > gate:
        return ("promoted",
                "Phase B is promoted as a retrieval candidate for further evaluation, "
                "not as a production system and not as SWE-bench evidence.")
    if delta >= TIE_BAND:
        return ("directional",
                "Phase B shows directional improvement but does not exceed the promotion gate.")
    if delta > -TIE_BAND:
        return ("tie",
                "Phase B tied within the observed flip-task noise band and is not promoted.")
    return ("null",
            "Phase B did not beat the stabilised existing dense baseline. The result is a "
            "null result for this embedder/backend under the current memory record format.")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_npass, base_stab, base_totals = load_baseline()
    baseline_mean = sum(base_totals) / len(base_totals) if base_totals else 0.0
    baseline_rng = (min(base_totals), max(base_totals)) if base_totals else (0, 0)
    gate = baseline_mean + NOISE_FLOOR
    n_tasks = len(base_npass)

    modes = {name: aggregate_mode(dirs) for name, dirs in MODE_RUN_DIRS.items()}
    if all(m["n_runs"] == 0 for m in modes.values()):
        print("No Phase B eval data found. Run the eval-v218-phaseb-* targets first.")
        sys.exit(1)

    for name, m in modes.items():
        if m["n_runs"] == 0:
            print(f"  WARNING: no data for mode '{name}' — run its eval targets.")
        else:
            print(f"  {name}: {m['n_runs']} runs  totals={m['totals']}  "
                  f"mean={m['mean']:.1f}/{n_tasks}")

    all_tasks = sorted(base_npass)

    # ── per-task matrix ──────────────────────────────────────────────────────
    matrix_rows = []
    for t in all_tasks:
        d_np = modes["dense"]["per_task"].get(t, "")
        h_np = modes["hybrid"]["per_task"].get(t, "")
        d_runs = modes["dense"]["n_runs"]
        h_runs = modes["hybrid"]["n_runs"]
        notes = []
        for label, np_, runs_ in (("dense", d_np, d_runs), ("hybrid", h_np, h_runs)):
            if runs_ and base_stab.get(t) == "stable_fail" and stability_label(np_, runs_) == "stable_pass":
                notes.append(f"{label}:stable_fail->pass")
            if runs_ and base_stab.get(t) == "stable_pass" and stability_label(np_, runs_) == "stable_fail":
                notes.append(f"{label}:stable_pass->fail(REGRESSION)")
        matrix_rows.append({
            "task_id": t,
            "family": family_of(t),
            "baseline_stability": base_stab.get(t, ""),
            "baseline_n_pass": base_npass.get(t, ""),
            "dense_n_pass": d_np,
            "hybrid_n_pass": h_np,
            "note": ";".join(notes),
        })

    matrix_path = OUT_DIR / "phase_b_per_task_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "family", "baseline_stability",
                                          "baseline_n_pass", "dense_n_pass",
                                          "hybrid_n_pass", "note"])
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"\nWrote {matrix_path}")

    # ── comparison CSV ───────────────────────────────────────────────────────
    comp_path = OUT_DIR / "phase_b_code_dense_comparison.csv"
    with open(comp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run1", "run2", "run3", "mean", "min", "max",
                    "delta_vs_baseline", "gate_pass"])
        w.writerow(["baseline_minilm_dense", base_totals[0], base_totals[1],
                    base_totals[2], f"{baseline_mean:.2f}", baseline_rng[0],
                    baseline_rng[1], "0.00", "n/a"])
        for name in ("dense", "hybrid"):
            m = modes[name]
            tt = m["totals"] + [""] * (3 - len(m["totals"]))
            delta = m["mean"] - baseline_mean if m["n_runs"] else 0.0
            gate_pass = (m["mean"] > gate) if m["n_runs"] else False
            w.writerow([f"code_{name}", tt[0], tt[1], tt[2],
                        f"{m['mean']:.2f}" if m["n_runs"] else "n/a",
                        m["range"][0] if m["n_runs"] else "",
                        m["range"][1] if m["n_runs"] else "",
                        f"{delta:+.2f}" if m["n_runs"] else "n/a",
                        str(gate_pass)])
    print(f"Wrote {comp_path}")

    # ── per-family breakdown ─────────────────────────────────────────────────
    families = sorted({family_of(t) for t in all_tasks})
    fam_lines = [
        "# v2.18 Phase B — Per-Family Breakdown",
        "",
        "Average tasks solved per family (n_pass summed over 3 runs ÷ 3). Families are",
        "derived from the task-id prefix token. Baseline = protected code-aware MiniLM",
        "dense (384d). Hybrid stage-1 shortlist is also code-aware MiniLM dense, not TF-IDF.",
        "",
        "| Family | Tasks | Baseline | Code-dense | Δ dense | Code-hybrid | Δ hybrid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in families:
        ftasks = [t for t in all_tasks if family_of(t) == fam]
        b = sum(base_npass[t] for t in ftasks) / 3
        d_runs = modes["dense"]["n_runs"]
        h_runs = modes["hybrid"]["n_runs"]
        d = (sum(modes["dense"]["per_task"].get(t, 0) for t in ftasks) / d_runs) if d_runs else None
        h = (sum(modes["hybrid"]["per_task"].get(t, 0) for t in ftasks) / h_runs) if h_runs else None
        d_str = f"{d:.2f}" if d is not None else "n/a"
        h_str = f"{h:.2f}" if h is not None else "n/a"
        dd = f"{d - b:+.2f}" if d is not None else "n/a"
        dh = f"{h - b:+.2f}" if h is not None else "n/a"
        fam_lines.append(f"| {fam} | {len(ftasks)} | {b:.2f} | {d_str} | {dd} | {h_str} | {dh} |")
    fam_path = OUT_DIR / "phase_b_per_family_breakdown.md"
    fam_path.write_text("\n".join(fam_lines) + "\n")
    print(f"Wrote {fam_path}")

    # ── verdict ──────────────────────────────────────────────────────────────
    ran = {n: m for n, m in modes.items() if m["n_runs"]}
    best_name = max(ran, key=lambda n: ran[n]["mean"]) if ran else None
    best = ran[best_name] if best_name else None
    strong_runs = {n: [t for t in m["totals"] if t >= 22] for n, m in ran.items()}
    conversions = [r["task_id"] for r in matrix_rows if "stable_fail->pass" in r["note"]]
    regressions = [r["task_id"] for r in matrix_rows if "REGRESSION" in r["note"]]

    if best:
        tag, sentence = verdict_for(best["mean"], baseline_mean, gate)
    else:
        tag, sentence = ("null", "No Phase B runs completed.")

    # ── summary.md ───────────────────────────────────────────────────────────
    s = [
        "# v2.18 Phase B — Code-Specialised Dense Retrieval Summary",
        "",
        f"**Baseline (Phase A, code-aware MiniLM dense, 384d):** mean {baseline_mean:.1f}/{n_tasks} "
        f"= {100*baseline_mean/n_tasks:.1f}%, range {baseline_rng[0]}–{baseline_rng[1]}.  ",
        f"**Promotion gate:** mean > {gate:.1f}/{n_tasks} across 3 runs (baseline mean + {NOISE_FLOOR:.0f} noise floor).",
        "",
        "## Results (clean 32-task benchmark, best-of-3)",
        "",
        "| Mode | run1 | run2 | run3 | mean | range | Δ vs baseline | gate |",
        "|---|---|---|---|---|---|---|---|",
        f"| Baseline MiniLM dense | {base_totals[0]} | {base_totals[1]} | {base_totals[2]} | "
        f"{baseline_mean:.1f} | {baseline_rng[0]}–{baseline_rng[1]} | — | — |",
    ]
    for name in ("dense", "hybrid"):
        m = modes[name]
        if m["n_runs"]:
            tt = m["totals"] + [""] * (3 - len(m["totals"]))
            delta = m["mean"] - baseline_mean
            s.append(f"| Code-{name} | {tt[0]} | {tt[1]} | {tt[2]} | {m['mean']:.1f} | "
                     f"{m['range'][0]}–{m['range'][1]} | {delta:+.1f} | "
                     f"{'PASS' if m['mean'] > gate else 'no'} |")
        else:
            s.append(f"| Code-{name} | — | — | — | n/a | n/a | n/a | not run |")

    s += [
        "",
        "## Verdict",
        "",
        f"**{tag.upper()}** — {sentence}",
        "",
    ]
    if best_name:
        s.append(f"Best Phase B mode: **code-{best_name}** (mean {best['mean']:.1f}/{n_tasks}, "
                 f"Δ {best['mean']-baseline_mean:+.1f} vs baseline).")
    s += [
        "",
        f"- Single-run ≥22/32 (strong): "
        + (", ".join(f"{n}={v}" for n, v in strong_runs.items() if v) or "none") + ".",
        f"- Stable-fail → stable-pass conversions: "
        + (", ".join(f"`{t}`" for t in conversions) or "none") + ".",
        f"- Stable-pass → stable-fail regressions: "
        + (", ".join(f"`{t}`" for t in regressions) or "none") + ".",
        "",
        "See `phase_b_code_dense_comparison.csv`, `phase_b_per_task_matrix.csv`,",
        "`phase_b_per_family_breakdown.md`, and `phase_b_claim_boundary.md`.",
    ]
    summary_path = OUT_DIR / "phase_b_code_dense_summary.md"
    summary_path.write_text("\n".join(s) + "\n")
    print(f"Wrote {summary_path}")

    # ── claim boundary ───────────────────────────────────────────────────────
    cb = [
        "# v2.18 Phase B — Claim Boundary",
        "",
        "## Verdict",
        "",
        f"**{tag.upper()}.** {sentence}",
        "",
        "## What this measures",
        "",
        f"- Encoder under test: a code-pretrained encoder (768d) vs the protected",
        f"  code-aware MiniLM dense baseline (384d, mean {baseline_mean:.1f}/{n_tasks}).",
        "- Clean 32-task benchmark, best-of-3, three runs per mode.",
        "- Hybrid stage-1 shortlist is code-aware MiniLM dense (not TF-IDF); stage-2 reranks",
        "  with the Phase B encoder.",
        "",
        "## Promotion gate (strict)",
        "",
        f"- Minimum candidate: mean > {gate:.1f}/{n_tasks} across three runs.",
        "- Strong result: ≥22/32 on a run. Stronger: converts ≥1 stable-fail task to",
        "  stable-pass without broad family-level regression.",
        "- A single high run is not enough. A 28-task improvement alone is not enough.",
        "- Single-family gain with other-family regression = family-specific, not promotion.",
        "",
        "## Not claimed (regardless of result)",
        "",
        "- No SWE-bench capability.",
        "- No production-grade reliability.",
        "- No AGI, quantum reasoning, or general superiority over other systems.",
        "- No claim that dense retrieval beats true TF-IDF (the baseline is already dense).",
        "- Results are bounded to the 32-task families tested at n=32, best-of-3.",
        "- No AI/tool/vendor attribution.",
    ]
    if tag != "promoted":
        cb += [
            "",
            "## Next direction (if not promoted)",
            "",
            "A larger or different encoder failing here does **not** mean dense retrieval",
            "fails globally. The likely bottleneck is the memory-record format: records are",
            "long ReAct trajectories mixing instruction, critique, tool calls, observations,",
            "errors, and recovery logic. Recommended next direction: operation-aware metadata",
            "or shorter memory summaries, evaluated against this same stabilised baseline.",
        ]
    cb_path = OUT_DIR / "phase_b_claim_boundary.md"
    cb_path.write_text("\n".join(cb) + "\n")
    print(f"Wrote {cb_path}")

    print(f"\nVerdict: {tag.upper()} — best mode "
          f"{best_name or 'n/a'} mean {best['mean']:.1f}/{n_tasks}" if best
          else f"\nVerdict: {tag.upper()}")


if __name__ == "__main__":
    main()
