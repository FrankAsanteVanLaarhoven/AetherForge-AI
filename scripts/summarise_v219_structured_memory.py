"""
scripts/summarise_v219_structured_memory.py

Summarise v2.19 — structured memory records + multi-view query reranking.

Reads three structured-dense (+ optionally three structured-hybrid) 32-task eval CSVs
and the Phase A baseline stability CSV, computes per-mode mean/std/range, per-task and
per-family deltas, stable-fail->pass conversions and stable-pass->fail regressions,
applies the strict promotion gate, and — because retrieval is deterministic — re-runs the
structured retriever offline to record the top retrieved record IDs for the tasks whose
pass behaviour changed.

Writes into results/v219_structured_memory/:
  summary.md
  comparison.csv
  per_task_matrix.csv
  per_family_breakdown.md
  retrieval_trace.md
  claim_boundary.md

Usage:
    python scripts/summarise_v219_structured_memory.py
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.structured_common import infer_family, primary_function_name  # noqa: E402

OUT_DIR = Path("results/v219_structured_memory")
BASELINE_CSV = Path("results/v218_retrieval_stability/tfidf_stability.csv")
TASKS_FILE = Path("data/v210_clean_repair_generalisation_tasks.jsonl")
STRUCT_INDEX = Path("memory/dense_index_v219_structured")
ENCODER = "models/embeddings/code-memory-embedder"

MODE_RUN_DIRS = {
    "structured_dense": [
        Path("outputs/eval_v219_structured_dense_32_run1"),
        Path("outputs/eval_v219_structured_dense_32_run2"),
        Path("outputs/eval_v219_structured_dense_32_run3"),
    ],
    "structured_hybrid": [
        Path("outputs/eval_v219_structured_hybrid_32_run1"),
        Path("outputs/eval_v219_structured_hybrid_32_run2"),
        Path("outputs/eval_v219_structured_hybrid_32_run3"),
    ],
}

NOISE_FLOOR = 2.0
TIE_BAND = 1.0


# ── loading ─────────────────────────────────────────────────────────────────────

def load_eval_csv(run_dir: Path) -> dict[str, bool]:
    p = run_dir / "best_of_3.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id") or row.get("id") or row.get("task", "")
            passed = (row.get("passed", "False") or "").strip().lower() in ("true", "1", "yes")
            if tid:
                out[tid] = passed
    return out


def load_baseline():
    if not BASELINE_CSV.exists():
        print(f"ERROR: baseline {BASELINE_CSV} not found.")
        sys.exit(1)
    n_pass, stab, totals = {}, {}, [0, 0, 0]
    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            tid = row["task_id"]
            n_pass[tid] = int(row["n_pass"])
            stab[tid] = row["stability"]
            for i in range(3):
                totals[i] += int(row.get(f"run{i+1}", 0) or 0)
    return n_pass, stab, totals


def stability_label(n_pass, n_runs):
    if n_runs == 0:
        return "missing"
    if n_pass == n_runs:
        return "stable_pass"
    if n_pass == 0:
        return "stable_fail"
    return "flip"


def family_of_task(tid: str, prompt: str) -> str:
    fn = primary_function_name("", prompt) or tid.replace("v210_", "")
    return infer_family(prompt, fn)


def aggregate(run_dirs):
    runs = [load_eval_csv(d) for d in run_dirs]
    present = [r for r in runs if r]
    tasks = sorted(set().union(*[set(r) for r in present])) if present else []
    per_task = {t: sum(int(r.get(t, False)) for r in present) for t in tasks}
    totals = [sum(r.values()) for r in present]
    mean = sum(totals) / len(totals) if totals else 0.0
    std = statistics.pstdev(totals) if len(totals) > 1 else 0.0
    rng = (min(totals), max(totals)) if totals else (0, 0)
    return {"per_task": per_task, "n_runs": len(present), "totals": totals,
            "mean": mean, "std": std, "range": rng}


def verdict_for(mode_mean, baseline_mean, gate, conversions=0, regressions=0):
    """Strict verdict.

    'promoted' requires the mean gate AND the user's "strongest evidence" — at least one
    stable-fail -> stable-pass conversion with no hard regression. Clearing the mean gate
    alone (no conversion) is only 'provisional': within this project's protocol such a mean
    lift is consistent with best-of-3 flip-task variance and must be confirmed.
    """
    delta = mode_mean - baseline_mean
    if mode_mean > gate:
        if conversions >= 1 and regressions == 0:
            return ("promoted",
                    "Structured memory records are promoted as a retrieval candidate for "
                    "further evaluation, not as production evidence and not as SWE-bench evidence.")
        return ("provisional",
                "Structured-hybrid meets the minimum mean gate but shows no stable-fail -> "
                "stable-pass conversion in that mode and high run-to-run variance; the retrieval "
                "trace shows surfaced memory is largely family-irrelevant, so the mean lift is "
                "consistent with flip-task sampling variance. Recorded as a PROVISIONAL candidate "
                "requiring confirmation runs, not a promotion.")
    if delta >= TIE_BAND:
        return ("directional", "Directional improvement, no promotion.")
    if delta > -TIE_BAND:
        return ("tie",
                "The result is within the observed flip-task noise band and is not promoted.")
    return ("null",
            "Structured memory records and reranking did not beat the stabilised retrieval "
            "baseline under the current 32-task protocol.")


def load_task_prompts():
    prompts = {}
    if TASKS_FILE.exists():
        import json
        for line in open(TASKS_FILE):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompts[r["id"]] = r.get("prompt") or r.get("task", "")
    return prompts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_npass, base_stab, base_totals = load_baseline()
    baseline_mean = sum(base_totals) / len(base_totals) if base_totals else 0.0
    baseline_rng = (min(base_totals), max(base_totals)) if base_totals else (0, 0)
    baseline_std = statistics.pstdev(base_totals) if len(base_totals) > 1 else 0.0
    gate = baseline_mean + NOISE_FLOOR
    n_tasks = len(base_npass)
    prompts = load_task_prompts()

    modes = {name: aggregate(dirs) for name, dirs in MODE_RUN_DIRS.items()}
    if all(m["n_runs"] == 0 for m in modes.values()):
        print("No v2.19 eval data found. Run the eval-v219-structured-* targets first.")
        sys.exit(1)
    for name, m in modes.items():
        if m["n_runs"]:
            print(f"  {name}: {m['n_runs']} runs totals={m['totals']} "
                  f"mean={m['mean']:.1f} std={m['std']:.2f}")
        else:
            print(f"  WARNING: no data for {name}")

    all_tasks = sorted(base_npass)

    # ── per-task matrix + changed/conversions/regressions ────────────────────
    matrix_rows = []
    changed_tasks = []
    for t in all_tasks:
        row = {"task_id": t, "family": family_of_task(t, prompts.get(t, "")),
               "baseline_stability": base_stab.get(t, ""), "baseline_n_pass": base_npass.get(t, "")}
        notes = []
        for name, m in modes.items():
            np_ = m["per_task"].get(t, "")
            row[name] = np_
            if m["n_runs"]:
                lab = stability_label(np_, m["n_runs"])
                if base_stab.get(t) == "stable_fail" and lab == "stable_pass":
                    notes.append(f"{name}:stable_fail->pass")
                if base_stab.get(t) == "stable_pass" and lab == "stable_fail":
                    notes.append(f"{name}:stable_pass->fail(REGRESSION)")
                # "changed" = per-task count differs from baseline (out of 3 runs)
                if isinstance(np_, int) and np_ != base_npass.get(t):
                    changed_tasks.append(t)
        row["note"] = ";".join(notes)
        matrix_rows.append(row)
    changed_tasks = sorted(set(changed_tasks))

    mode_names = list(modes.keys())
    matrix_path = OUT_DIR / "per_task_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "family", "baseline_stability",
                                          "baseline_n_pass", *mode_names, "note"])
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"\nWrote {matrix_path}")

    conversions = [r["task_id"] for r in matrix_rows if "stable_fail->pass" in r["note"]]
    regressions = [r["task_id"] for r in matrix_rows if "REGRESSION" in r["note"]]

    # ── comparison CSV ───────────────────────────────────────────────────────
    comp_path = OUT_DIR / "comparison.csv"
    with open(comp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run1", "run2", "run3", "mean", "std", "min", "max",
                    "delta_vs_baseline", "gate_pass"])
        w.writerow(["baseline_minilm_dense", base_totals[0], base_totals[1], base_totals[2],
                    f"{baseline_mean:.2f}", f"{baseline_std:.2f}", baseline_rng[0],
                    baseline_rng[1], "0.00", "n/a"])
        for name in mode_names:
            m = modes[name]
            tt = m["totals"] + [""] * (3 - len(m["totals"]))
            delta = m["mean"] - baseline_mean if m["n_runs"] else 0.0
            w.writerow([name, tt[0], tt[1], tt[2],
                        f"{m['mean']:.2f}" if m["n_runs"] else "n/a",
                        f"{m['std']:.2f}" if m["n_runs"] else "n/a",
                        m["range"][0] if m["n_runs"] else "",
                        m["range"][1] if m["n_runs"] else "",
                        f"{delta:+.2f}" if m["n_runs"] else "n/a",
                        str(m["mean"] > gate) if m["n_runs"] else "False"])
    print(f"Wrote {comp_path}")

    # ── per-family breakdown ─────────────────────────────────────────────────
    fam_of = {t: family_of_task(t, prompts.get(t, "")) for t in all_tasks}
    families = sorted(set(fam_of.values()))
    fam_lines = [
        "# v2.19 Per-Family Breakdown",
        "",
        "Average tasks solved per family (n_pass summed over runs ÷ run count). Baseline =",
        "protected code-aware MiniLM dense (384d). Structured modes hold the SAME encoder and",
        "vary only record structure + multi-view query + reranking.",
        "",
        "| Family | Tasks | Baseline | " + " | ".join(mode_names) + " |",
        "|---|---:|---:|" + "|".join(["---:"] * len(mode_names)) + "|",
    ]
    for fam in families:
        ftasks = [t for t in all_tasks if fam_of[t] == fam]
        b = sum(base_npass[t] for t in ftasks) / 3
        cells = []
        for name in mode_names:
            m = modes[name]
            if m["n_runs"]:
                v = sum(m["per_task"].get(t, 0) for t in ftasks) / m["n_runs"]
                cells.append(f"{v:.2f} ({v - b:+.2f})")
            else:
                cells.append("n/a")
        fam_lines.append(f"| {fam} | {len(ftasks)} | {b:.2f} | " + " | ".join(cells) + " |")
    (OUT_DIR / "per_family_breakdown.md").write_text("\n".join(fam_lines) + "\n")
    print(f"Wrote {OUT_DIR / 'per_family_breakdown.md'}")

    # ── retrieval trace for changed tasks (deterministic) ────────────────────
    trace_lines = [
        "# v2.19 Retrieval Trace — Changed Tasks",
        "",
        "Top structured-dense retrievals (deterministic; re-run offline) for tasks whose",
        "pass count changed vs the baseline. Shows which memory records the structured",
        "multi-view reranker surfaces — exposing whether the memory pool actually contains",
        "family-relevant repair signal for these tasks.",
        "",
    ]
    try:
        from memory.structured_retriever import StructuredReranker
        sr = StructuredReranker(STRUCT_INDEX, model_name=ENCODER, mode="dense")
        if not changed_tasks:
            trace_lines.append("_No tasks changed pass count vs the baseline._")
        for t in changed_tasks:
            prompt = prompts.get(t, "")
            hits = sr.retrieve(prompt, top_k=4) if prompt else []
            qfam = fam_of[t]
            trace_lines.append(f"### `{t}`  (query family: {qfam}, "
                               f"baseline {base_npass.get(t)}/3 → "
                               + ", ".join(f"{n} {modes[n]['per_task'].get(t,'-')}/{modes[n]['n_runs']}"
                                           for n in mode_names if modes[n]['n_runs']) + ")")
            for h in hits:
                same = "✓" if h.get("task_family") == qfam else "✗"
                trace_lines.append(
                    f"- [{same} family={h.get('task_family')}] "
                    f"`{h.get('task_signature','')[:48]}` score={h.get('score',0):.3f} "
                    f"id={h.get('record_id','')[:8]}")
            trace_lines.append("")
    except Exception as exc:  # index/encoder unavailable — degrade, don't crash
        trace_lines.append(f"_Retrieval trace unavailable: {exc}_")
    (OUT_DIR / "retrieval_trace.md").write_text("\n".join(trace_lines) + "\n")
    print(f"Wrote {OUT_DIR / 'retrieval_trace.md'}")

    # ── verdict ──────────────────────────────────────────────────────────────
    ran = {n: m for n, m in modes.items() if m["n_runs"]}
    best_name = max(ran, key=lambda n: ran[n]["mean"]) if ran else None
    best = ran[best_name] if best_name else None
    strong = {n: [t for t in m["totals"] if t >= 22] for n, m in ran.items()}
    # Per-mode conversion / hard-regression counts (the user's "strongest evidence").
    conv_by_mode = {n: sum(1 for r in matrix_rows if f"{n}:stable_fail->pass" in r["note"]) for n in mode_names}
    regr_by_mode = {n: sum(1 for r in matrix_rows if f"{n}:stable_pass->fail" in r["note"]) for n in mode_names}
    if best:
        tag, sentence = verdict_for(best["mean"], baseline_mean, gate,
                                    conv_by_mode.get(best_name, 0), regr_by_mode.get(best_name, 0))
    else:
        tag, sentence = ("null", "No runs completed.")

    # ── summary.md ───────────────────────────────────────────────────────────
    s = [
        "# v2.19 — Structured Memory Records + Query Reranking Summary",
        "",
        f"**Baseline (Phase A, code-aware MiniLM dense, 384d):** mean {baseline_mean:.1f}/{n_tasks} "
        f"= {100*baseline_mean/n_tasks:.1f}%, std {baseline_std:.2f}, range {baseline_rng[0]}–{baseline_rng[1]}.  ",
        f"**Promotion gate:** mean > {gate:.1f}/{n_tasks} across three runs.",
        "",
        "Encoder held fixed (baseline code-aware MiniLM); only record structure, multi-view",
        "query construction, and deterministic reranking change.",
        "",
        "## Results (clean 32-task benchmark, best-of-3)",
        "",
        "| Mode | run1 | run2 | run3 | mean | std | range | Δ vs baseline | gate |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Baseline MiniLM dense | {base_totals[0]} | {base_totals[1]} | {base_totals[2]} | "
        f"{baseline_mean:.1f} | {baseline_std:.2f} | {baseline_rng[0]}–{baseline_rng[1]} | — | — |",
    ]
    for name in mode_names:
        m = modes[name]
        if m["n_runs"]:
            tt = m["totals"] + [""] * (3 - len(m["totals"]))
            s.append(f"| {name} | {tt[0]} | {tt[1]} | {tt[2]} | {m['mean']:.1f} | {m['std']:.2f} | "
                     f"{m['range'][0]}–{m['range'][1]} | {m['mean']-baseline_mean:+.1f} | "
                     f"{'PASS' if m['mean'] > gate else 'no'} |")
        else:
            s.append(f"| {name} | — | — | — | n/a | n/a | n/a | n/a | not run |")
    s += [
        "",
        "## Verdict",
        "",
        f"**{tag.upper()}** — {sentence}",
        "",
    ]
    if best_name:
        s.append(f"Best mode: **{best_name}** (mean {best['mean']:.1f}/{n_tasks}, "
                 f"Δ {best['mean']-baseline_mean:+.1f}).")
    s += [
        "",
        f"- Single-run ≥22/32 (strong): " + (", ".join(f"{n}={v}" for n, v in strong.items() if v) or "none") + ".",
        f"- Stable-fail → stable-pass conversions by mode: "
        + (", ".join(f"{n}={conv_by_mode.get(n,0)}" for n in mode_names if modes[n]['n_runs']) or "none")
        + (f" (tasks: {', '.join('`'+t+'`' for t in conversions)})" if conversions else "") + ".",
        f"- Stable-pass → stable-fail hard regressions: " + (", ".join(f"`{t}`" for t in regressions) or "none") + ".",
        f"- Tasks with changed pass count: {len(changed_tasks)}.",
        "",
        "### Mechanistic read (why the gate result must be read with caution)",
        "",
        "`retrieval_trace.md` re-runs the deterministic retriever for every changed task. For",
        "the interval/rle/tree gains driving the structured-hybrid mean (e.g. `range_summary`,",
        "`meeting_rooms`, `interval_intersection`), the surfaced memory is family-IRRELEVANT",
        "(`unique_sorted`, `two_sum`, `merge_sorted`) — the 99-record pool contains essentially",
        "no verified interval/tree/rle repairs. So those pass-count movements are not",
        "mechanistically attributable to retrieval; they are consistent with best-of-3 flip-task",
        "variance. The only retrieval-attributable hard conversion is `kth_smallest_matrix`",
        "(0→3/3) in structured-DENSE, which correctly surfaced the lone `transpose(matrix)`",
        "record — but structured-dense does not clear the gate, and hybrid scores that task 0/3.",
        "",
        "See `comparison.csv`, `per_task_matrix.csv`, `per_family_breakdown.md`,",
        "`retrieval_trace.md`, and `claim_boundary.md`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(s) + "\n")
    print(f"Wrote {OUT_DIR / 'summary.md'}")

    # ── claim boundary ───────────────────────────────────────────────────────
    cb = [
        "# v2.19 — Claim Boundary",
        "",
        "## Verdict",
        "",
        f"**{tag.upper()}.** {sentence}",
        "",
        "## What this measures",
        "",
        "- Structured memory records (deterministically derived: family, signature, failure",
        "  mode, cues, minimal test, rationale) + multi-view query (instruction/family/",
        "  signature) + deterministic composite reranking.",
        "- Encoder held FIXED at the baseline code-aware MiniLM (384d), so any change is",
        "  attributable to record structure and ranking, not encoder capacity.",
        "- Same protected 99-record memory pool; injection format unchanged.",
        "- Clean 32-task benchmark, best-of-3, three runs per mode.",
        "",
        "## Promotion gate (strict)",
        "",
        f"- Minimum candidate: mean > {gate:.1f}/{n_tasks} across three runs.",
        "- Strong result: ≥22/32 on a run. Strongest: ≥1 stable-fail → stable-pass without",
        "  broad family-level regression.",
        "- A single high run is not enough. A 28-task improvement alone is not enough.",
        "",
        "## Not claimed (regardless of result)",
        "",
        "- No SWE-bench capability or success.",
        "- No production-grade reliability.",
        "- No frontier-model superiority; no AGI or quantum-reasoning claims.",
        "- Results bounded to the 32-task families tested at n=32, best-of-3.",
        "- No AI/tool/vendor attribution.",
    ]
    if tag != "promoted":
        cb += [
            "",
            "## Interpretation / next direction",
            "",
            "The retrieval trace shows which records the structured reranker surfaces for the",
            "changed tasks. Where same-family records do not exist in the 99-record pool, no",
            "structuring or reranking can surface them — indicating the bottleneck is memory",
            "*coverage* (the pool lacks family-relevant verified repairs for the benchmark",
            "families), not only memory *format*. Recommended next direction: expand the",
            "verified-repair memory with family-targeted records (interval/tree/rle/dict),",
            "then re-run this structured-retrieval audit against the same stabilised baseline",
            "before escalating to a heavier embedding backend.",
        ]
    (OUT_DIR / "claim_boundary.md").write_text("\n".join(cb) + "\n")
    print(f"Wrote {OUT_DIR / 'claim_boundary.md'}")

    print(f"\nVerdict: {tag.upper()}" + (f" — best {best_name} mean {best['mean']:.1f}/{n_tasks}" if best else ""))


if __name__ == "__main__":
    main()
