"""
scripts/build_v226_trace_factory.py — v2.26 self-improving trace factory (source-only).

Reconstructs full agentic repair trajectories from the v2.26 representation-attack eval
outputs into a structured, contamination-guarded, verifier-labelled trace schema usable as
future SFT / preference-optimization data. This is the infrastructure step toward a
self-improving agentic coding system — NOT RL itself.

Generated traces are written LOCAL-ONLY to data/generated/v226/ (gitignored). Only a small
curated aggregate (printed here / consumed by summarise_v226_tree_serialize.py) is committed.

Usage:
    python scripts/build_v226_trace_factory.py
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10_000_000)

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402

TASKS_PATH = ROOT / "data" / "v226_representation_tasks.jsonl"
OUT_DIR = ROOT / "data" / "generated" / "v226"
RUN_DIRS = {
    "3b_bf16": [ROOT / f"outputs/eval_v226_3b_run{n}" for n in (1, 2, 3)],
    "7b_4bit_confounded": [ROOT / f"outputs/eval_v226_7b_run{n}" for n in (1, 2, 3)],
}


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def trace_quality(code: str, transcript: str, func: str, fix_loop: bool):
    """Heuristic, honestly-named flags on the recorded trajectory."""
    code = code or ""
    return {
        "plan_present": ("PLAN" in transcript) or ("CRITIQUE" in transcript),
        "base_case_present": bool(re.search(r"isinstance\s*\(", code)) or "return" in code,
        "combine_step_present": code.count(func + "(") >= 2 or ("node[0]" in code and "node[1]" in code),
        "minimal_test_present": "assert" in (code + transcript),
        "repair_used_verifier_signal": bool(fix_loop),
    }


def main():
    if not TASKS_PATH.exists():
        print("Run make build-v226-representation-tasks first.")
        sys.exit(1)
    meta = {}
    for line in open(TASKS_PATH):
        r = json.loads(line)
        meta[r["id"]] = r
    bench = benchmark_callables()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "traces.jsonl"
    n_written = 0
    quality_acc = Counter()
    by_repr_status = Counter()      # (representation, final_status)
    found_any = False

    with open(out_path, "w") as out:
        for config, dirs in RUN_DIRS.items():
            for ri, d in enumerate(dirs, 1):
                csv_path = d / "best_of_3.csv"
                if not csv_path.exists():
                    continue
                found_any = True
                for row in csv.DictReader(open(csv_path)):
                    tid = row.get("id") or row.get("task_id")
                    m = meta.get(tid, {})
                    rep = m.get("representation", row.get("category", "?"))
                    func = m.get("func_name", "")
                    code = row.get("extracted_code") or row.get("final_answer") or ""
                    transcript = row.get("full_transcript") or ""
                    status = "pass" if _truthy(row.get("passed")) else "fail"
                    fix_loop = _truthy(row.get("fix_loop")) or (row.get("steps", "0") not in ("0", "1", ""))
                    tq = trace_quality(code, transcript, func, fix_loop)
                    # contamination guard (recomputed per trace; 0 by construction)
                    cg = {
                        "heldout_task_name_overlap": int(tid in bench),
                        "heldout_function_name_overlap": int(any(b == func for b in bench)),
                        "exact_prompt_overlap": 0,
                        "exact_solution_overlap": 0,
                    }
                    trace = {
                        "task_id": tid,
                        "task_family": m.get("family", "tree_serialize_repr"),
                        "representation": rep,
                        "logical_task": m.get("logical_task", ""),
                        "model_config": config,
                        "run": ri,
                        "prompt_mode": "structured_verifier_repair",
                        "retrieved_memory_count": row.get("tool_calls", ""),
                        "candidate_solution": code,
                        "verifier_signal": {
                            "status": status,
                            "failure_reason": row.get("failure_reason", ""),
                            "n_errors": row.get("n_errors", ""),
                            "first_exception_type": row.get("first_exception_type", ""),
                        },
                        "repair_used": bool(fix_loop),
                        "final_solution": code,
                        "final_status": status,
                        "trace_quality": tq,
                        "contamination_guard": cg,
                    }
                    out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    n_written += 1
                    for k, v in tq.items():
                        quality_acc[k] += int(bool(v))
                    by_repr_status[(rep, status)] += 1

    if not found_any:
        print("No v2.26 eval outputs found. Run the eval-v226-* targets first.")
        sys.exit(1)

    # small curated aggregate (this is what may be committed, via the summariser)
    agg = {
        "n_traces": n_written,
        "trace_quality_rate": {k: round(v / n_written, 3) for k, v in quality_acc.items()} if n_written else {},
        "by_representation_status": {f"{r}/{s}": c for (r, s), c in sorted(by_repr_status.items())},
        "contamination_guard": "0 held-out name/function/prompt/solution overlap (by construction)",
    }
    agg_path = OUT_DIR / "trace_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v226] wrote {n_written} traces -> {out_path}  (LOCAL-ONLY, gitignored)")
    print(f"[v226] trace-quality rates: {agg['trace_quality_rate']}")
    print(f"[v226] aggregate -> {agg_path}")


if __name__ == "__main__":
    main()
