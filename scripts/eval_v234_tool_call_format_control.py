"""
scripts/eval_v234_tool_call_format_control.py — v2.34 tool-call format-control evaluation.

Inference-time control experiment (NO training, NO new generation). Takes the v2.33 frozen-32-task
benchmark transcripts (the scaffold adapter WITHOUT tool-call control) and applies the deterministic
tool_call_format_control controller to each task's model output, measuring how much valid
execute_code / tool-call emission is recovered, and whether any passes are recovered when the
re-wrapped code is genuinely runnable. Compares:

    baseline  : v2.33 scaffold adapter, no tool-call control
    controlled: same outputs, with deterministic tool-call format control

Recovered pass is counted ONLY when the re-wrapped code defines the task function and executes to
PASS (the controller never invents code), so improvements are real, not fabricated.

Output (LOCAL-ONLY, gitignored): outputs/v234_tool_call_format_control/benchmark_metrics.json

Usage:
    python scripts/eval_v234_tool_call_format_control.py [--baseline-csv <path>]
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10_000_000)

from scripts.tool_call_format_control import (  # noqa: E402
    detect_tool_call, has_invalid_tool_json, repair_to_execute_code,
)

BASELINE_CSV = ROOT / "outputs/v233_scaffold_first_sft/eval_32task_adapter/best_of_3.csv"
TASKS = ROOT / "data" / "v210_clean_repair_generalisation_tasks.jsonl"
OUT_DIR = ROOT / "outputs" / "v234_tool_call_format_control"
HARD_TREE_IDS = ("tree_serialize", "tree_from_list", "tree_max_path_sum")
CHAMPION_32 = 23
V233_BASE_PASS = 5   # v2.33 frozen-32 adapter pass count (the bar v2.34 must beat)


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _task_index():
    """id -> (func_name, [task test assertions parsed from the prompt's `Verify:` clause])."""
    idx = {}
    if not TASKS.exists():
        return idx
    for line in open(TASKS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prompt = r.get("prompt") or r.get("task", "")
        fm = re.search(r"`([a-zA-Z_]\w*)\s*\(", prompt)
        func = fm.group(1) if fm else ""
        tests = []
        vm = re.search(r"Verify:\s*(.*)", prompt, re.DOTALL)
        if vm and func:
            body = vm.group(1).strip().rstrip(".")
            # split into per-call chunks (each real test starts with `func(`)
            for chunk in re.split(r"(?=\b" + re.escape(func) + r"\s*\()", body):
                c = chunk.strip().rstrip(",").strip().rstrip(".").strip().rstrip(",").strip()
                if c.startswith(func) and "==" in c:
                    tests.append(c)
        idx[r["id"]] = (func, tests)
    return idx


def _strip_model_verification(code):
    """Keep function/helper defs and their bodies; drop the model's own assert/print lines."""
    out = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if s.startswith("assert ") or s.startswith("print(") or s.startswith("# Test"):
            continue
        out.append(ln)
    return "\n".join(out)


def _verify_against_task(code, func, tests):
    """True ONLY if the re-wrapped code defines `func` and passes the TASK's real assertions
    (the model's own print('PASS')/asserts are stripped — never trusted)."""
    if not code or f"def {func}" not in code or not tests:
        return False
    body = _strip_model_verification(code)
    prog = body + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"; fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and "PASS" in r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", default=str(BASELINE_CSV))
    args = ap.parse_args()
    csv_path = Path(args.baseline_csv)
    if not csv_path.exists():
        print(f"[v234] baseline benchmark CSV not found: {csv_path}\n"
              "Run the v2.33 GPU runbook first (make v233-gpu-runbook).")
        sys.exit(1)
    task_idx = _task_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(csv_path)))
    base_pass = ctrl_pass = base_tool = ctrl_tool = exec_code = 0
    no_tool_base = no_tool_ctrl = invalid_json = 0
    recovered_calls = recovered_passes = unsafe_rejected = 0
    base_reasons, ctrl_reasons = Counter(), Counter()
    hard = {"base": [0, 0], "ctrl": [0, 0]}
    ts = {"base": [0, 0], "ctrl": [0, 0]}

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
        if rep["status"] == "rejected":
            unsafe_rejected += 1
        recovered = False
        if not base_made_call and rep["status"] == "ok":
            recovered_calls += 1
            if _verify_against_task(rep.get("code"), func, tests):
                recovered = True
                recovered_passes += 1
        ctrl_passed = base_passed or recovered

        base_pass += int(base_passed); ctrl_pass += int(ctrl_passed)
        base_tool += int(base_made_call); ctrl_tool += int(ctrl_made_call)
        exec_code += int(det["has_tool_call"] and det["kind"] == "execute_code")
        no_tool_base += int(not base_made_call)
        no_tool_ctrl += int(not ctrl_made_call)
        invalid_json += int(has_invalid_tool_json(gen))
        if not base_passed:
            base_reasons[(r.get("failure_reason") or ("no_tool_call" if not base_made_call else "other")).strip()] += 1
        if not ctrl_passed:
            if not ctrl_made_call:
                ctrl_reasons["no_tool_call"] += 1            # control could not recover a call
            elif not base_made_call:
                ctrl_reasons["wrapped_no_passing_solution"] += 1  # recovered a call, but code has no passing solution
            else:
                ctrl_reasons[(r.get("failure_reason") or "other").strip()] += 1
        if any(h in tid for h in HARD_TREE_IDS):
            hard["base"][0] += int(base_passed); hard["base"][1] += 1
            hard["ctrl"][0] += int(ctrl_passed); hard["ctrl"][1] += 1
        if "tree_serialize" in tid:
            ts["base"][0] += int(base_passed); ts["base"][1] += 1
            ts["ctrl"][0] += int(ctrl_passed); ts["ctrl"][1] += 1

    n = len(rows)
    metrics = {
        "n": n, "champion_32_pass": CHAMPION_32, "v233_baseline_pass": V233_BASE_PASS,
        "baseline": {
            "pass": base_pass, "tool_call_rate": round(base_tool / n, 3) if n else 0.0,
            "no_tool_call": no_tool_base, "failure_reasons": dict(base_reasons),
        },
        "controlled": {
            "pass": ctrl_pass, "tool_call_rate": round(ctrl_tool / n, 3) if n else 0.0,
            "execute_code_rate": round(exec_code / n, 3) if n else 0.0,
            "no_tool_call": no_tool_ctrl, "invalid_tool_json": invalid_json,
            "failure_reasons": dict(ctrl_reasons),
        },
        "recovered_tool_calls": recovered_calls, "recovered_passes": recovered_passes,
        "unsafe_or_ambiguous_rejected": unsafe_rejected,
        "no_tool_call_dominant_controlled": bool(ctrl_reasons) and ctrl_reasons.most_common(1)[0][0] == "no_tool_call",
        "hard_tree_baseline": f"{hard['base'][0]}/{hard['base'][1]}",
        "hard_tree_controlled": f"{hard['ctrl'][0]}/{hard['ctrl'][1]}",
        "tree_serialize_preserved": bool(ts["ctrl"][1] == 0 or ts["ctrl"][0] >= ts["base"][0]),
        "tree_serialize_controlled": f"{ts['ctrl'][0]}/{ts['ctrl'][1]}",
    }
    (OUT_DIR / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    b, c = metrics["baseline"], metrics["controlled"]
    print(f"[v234] baseline pass {b['pass']}/{n} tool_call_rate {b['tool_call_rate']} no_tool_call {b['no_tool_call']}")
    print(f"[v234] controlled pass {c['pass']}/{n} tool_call_rate {c['tool_call_rate']} no_tool_call {c['no_tool_call']} "
          f"(dominant={metrics['no_tool_call_dominant_controlled']})")
    print(f"[v234] recovered tool-calls {recovered_calls}, recovered passes {recovered_passes}, "
          f"rejected {unsafe_rejected}; tree_serialize {metrics['tree_serialize_controlled']}")


if __name__ == "__main__":
    main()
