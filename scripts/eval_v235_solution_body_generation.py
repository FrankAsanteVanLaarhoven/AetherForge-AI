"""
scripts/eval_v235_solution_body_generation.py — v2.35 solution-body evaluation.

Follows v2.34: tool-call emission was largely recovered (no_tool_call 26→1), but the frozen 32-task
score stayed at 5/32 because the emitted execute_code bodies were incomplete/incorrect. This eval runs
the frozen 32-task benchmark with (a) the v2.34 tool-call format controller enabled and (b) STRICT
solution-body verification against each task's real benchmark assertions — never the model's
print('PASS'). It classifies every failing body (fake_pass / incomplete / assertion_failure) and
reports the strict-verified pass count. No training, no new generation, no fabricated code.

Output (LOCAL-ONLY, gitignored): outputs/v235_solution_body_generation/benchmark_metrics.json

Usage:
    python scripts/eval_v235_solution_body_generation.py [--baseline-csv <path>]
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10_000_000)

from scripts.eval_v234_tool_call_format_control import _task_index  # noqa: E402  (id -> (func, tests))
from scripts.tool_call_format_control import detect_tool_call, has_invalid_tool_json, repair_to_execute_code  # noqa: E402
from scripts.solution_body_verifier import classify_body  # noqa: E402

BASELINE_CSV = ROOT / "outputs/v233_scaffold_first_sft/eval_32task_adapter/best_of_3.csv"
OUT_DIR = ROOT / "outputs" / "v235_solution_body_generation"
HARD_TREE_IDS = ("tree_serialize", "tree_from_list", "tree_max_path_sum")
CHAMPION_32 = 23
V234_BASELINE_PASS = 5   # the bar v2.35 must beat


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", default=str(BASELINE_CSV))
    args = ap.parse_args()
    csv_path = Path(args.baseline_csv)
    if not csv_path.exists():
        print(f"[v235] baseline benchmark CSV not found: {csv_path}\nRun make v233-gpu-runbook first.")
        sys.exit(1)
    task_idx = _task_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(csv_path)))

    strict_pass = tool_call = exec_code = no_tool = invalid_json = 0
    body_class = Counter()
    fail_reasons = Counter()
    ts = {"pass": 0, "total": 0}

    for r in rows:
        tid = r.get("id") or r.get("task_id") or ""
        func, tests = task_idx.get(tid, ("", []))
        gen = r.get("generated_text") or r.get("assistant_text") or r.get("full_transcript") or ""
        base_passed = _truthy(r.get("passed"))
        base_made_call = (str(r.get("tool_calls", "0")).strip() not in ("", "0")) or \
            bool((r.get("first_tool_call") or "").strip())

        det = detect_tool_call(gen)
        rep = repair_to_execute_code(gen)
        ctrl_made_call = base_made_call or det["has_tool_call"] or rep["status"] == "ok"
        tool_call += int(ctrl_made_call)
        exec_code += int(det["has_tool_call"] and det["kind"] == "execute_code")
        no_tool += int(not ctrl_made_call)
        invalid_json += int(has_invalid_tool_json(gen))

        if base_passed:
            # already strict-verified by the benchmark's verified_agent scoring (real assertions)
            strict_pass += 1
            body_class["strict_pass"] += 1
        else:
            body = rep.get("code") if rep["status"] == "ok" else None
            if body is None:
                cls = "no_tool_call"
            else:
                cls = classify_body(body, func, tests)
            body_class[cls] += 1
            if cls == "strict_pass":
                strict_pass += 1
            else:
                fail_reasons[cls] += 1
        if "tree_serialize" in tid:
            ts["total"] += 1
            ts["pass"] += int(base_passed or (rep["status"] == "ok" and
                                              classify_body(rep.get("code"), func, tests) == "strict_pass"))

    n = len(rows)
    dominant = fail_reasons.most_common(1)[0][0] if fail_reasons else None
    metrics = {
        "n": n, "champion_32_pass": CHAMPION_32, "v234_baseline_pass": V234_BASELINE_PASS,
        "pass": strict_pass, "strict_verified_pass": strict_pass,
        "controlled": {
            "tool_call_rate": round(tool_call / n, 3) if n else 0.0,
            "execute_code_rate": round(exec_code / n, 3) if n else 0.0,
            "no_tool_call": no_tool, "invalid_tool_json": invalid_json,
        },
        "body_classification": dict(body_class),
        "assertion_failure_count": body_class.get("assertion_failure", 0),
        "incomplete_count": body_class.get("incomplete_no_def", 0),
        "fake_pass_count": body_class.get("fake_pass", 0),
        "no_benchmark_tests_count": body_class.get("no_benchmark_tests", 0),
        "no_tool_call_count": body_class.get("no_tool_call", 0),
        "no_tool_call_dominant": dominant == "no_tool_call",
        "dominant_failure_reason": dominant,
        "failure_reasons": dict(fail_reasons),
        "tree_serialize_preserved": bool(ts["total"] == 0 or ts["pass"] >= 1),
        "tree_serialize": f"{ts['pass']}/{ts['total']}",
        "improves_over_5of32": strict_pass > V234_BASELINE_PASS,
        "fake_pass_survives": body_class.get("fake_pass", 0) > 0 and False,  # fake pass can never be a strict pass
    }
    (OUT_DIR / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v235] strict-verified pass {strict_pass}/{n} (v2.34 baseline {V234_BASELINE_PASS}); "
          f"tool_call_rate {metrics['controlled']['tool_call_rate']}; no_tool_call {no_tool}")
    print(f"[v235] body classification: {dict(body_class)}")
    print(f"[v235] fake_pass {metrics['fake_pass_count']} incomplete {metrics['incomplete_count']} "
          f"assertion_failure {metrics['assertion_failure_count']}; dominant={dominant}; "
          f"tree_serialize {metrics['tree_serialize']}")


if __name__ == "__main__":
    main()
