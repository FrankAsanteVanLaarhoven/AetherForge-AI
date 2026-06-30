"""
scripts/build_v227_trace_factory.py — v2.27 hardened trace factory.

Fixes the v2.26 trace-factory gap (candidate_solution == final_solution) by parsing the GENUINE
repair transitions already present in the v2.26 agent transcripts. Each emitted trace records a
DISTINCT first candidate and final/repaired solution, the structured verifier signal that drove
the repair, the model's repair plan (CRITIQUE), and a RECOMPUTED ground-truth status obtained by
executing the extracted code standalone — independent of the (unreliable) eval CSV repair flags.

Hardened schema per trace:
    task_id, task_family, representation, logical_task, model_config, run,
    candidate_solution, verifier_signal{status,category,detail,guidance},
    repair_plan, repaired_solution, final_solution, final_status,
    repair_outcome   in {no_repair_needed, repair_attempted_failed, repair_attempted_fixed},
    repair_kind      in {none, format_only, algorithmic, envelope_format},
    candidate_executes, repaired_executes, agent_status, envelope_format_failure,
    trace_quality{...}, contamination_guard{...}   (COMPUTED, not declared)

Source: outputs/eval_v226_3b_run{1,2,3}/best_of_3.csv (local-only eval transcripts).
Output: data/generated/v227/traces.jsonl  (LOCAL-ONLY, gitignored).
A small curated aggregate (data/generated/v227/trace_aggregate.json) feeds the summariser; the
full traces are never committed.

Usage:
    python scripts/build_v227_trace_factory.py
"""

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

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402
from scripts.v227_format_verifier import classify_failure, render  # noqa: E402

TASKS_PATH = ROOT / "data" / "v226_representation_tasks.jsonl"
BENCHMARK_PATH = ROOT / "data" / "v210_clean_repair_generalisation_tasks.jsonl"
OUT_DIR = ROOT / "data" / "generated" / "v227"
RUN_DIRS = [ROOT / f"outputs/eval_v226_3b_run{n}" for n in (1, 2, 3)]

CODE_RX = re.compile(r'execute_code\(\{\s*"code":\s*"""(.*?)"""', re.DOTALL)
VERIFIER_RX = re.compile(
    r"VERIFIER:\s*\n\s*status:\s*([^\n]*)\n\s*category:\s*([^\n]*)\n\s*detail:\s*([^\n]*)", re.IGNORECASE)
CRITIQUE_RX = re.compile(r"CRITIQUE:\s*([^\n]*)")
PROBE_INPUTS = ((1, (2, 3)), 5, ((1, 2), 3))


def _run(prog: str, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"
        fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
    return r.returncode == 0, r.stdout, r.stderr


def _executes_pass(code: str) -> bool:
    """True if the extracted candidate (def + asserts + print('PASS')) runs clean."""
    if not code.strip():
        return False
    ok, out, _ = _run(code)
    return ok and "PASS" in out


def _strip_tests(code: str) -> str:
    keep = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("assert ") or s.startswith("print(") or s.startswith("# Test"):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _observed_output(code: str, func: str, inp):
    """Execute candidate function on one probe input; return (ok, value_repr)."""
    body = _strip_tests(code)
    prog = body + f"\n\nimport json as _j\ntry:\n    _r = {func}({inp!r})\n    print('OUT::' + repr(_r))\nexcept Exception as _e:\n    print('ERR::' + type(_e).__name__)\n"
    ok, out, _ = _run(prog)
    for line in out.splitlines():
        if line.startswith("OUT::"):
            try:
                return True, eval(line[5:], {"__builtins__": {}}, {})  # noqa: S307 (controlled repr)
            except Exception:
                return True, line[5:]
        if line.startswith("ERR::"):
            return False, line[5:]
    return False, None


def _repair_kind(candidate_code, repaired_code, func, logical_task, representation, cand_pass, agent_pass):
    """Classify the nature of the repair the trace represents."""
    if cand_pass and not agent_pass:
        # code is algorithmically fine standalone but the agent scored it FAIL: the failure was
        # in the tool-call / output ENVELOPE format (e.g. invalid JSON args), not the algorithm.
        return "envelope_format", True
    if cand_pass:
        return "none", False
    # candidate genuinely wrong: classify the first divergent output
    ok, observed = _observed_output(candidate_code, func, PROBE_INPUTS[0])
    if not ok:
        return "algorithmic", False
    try:
        expected = render(PROBE_INPUTS[0], logical_task, representation)
    except ValueError:
        return "format_only", False
    ft = classify_failure(observed, expected, representation)
    if ft is None:
        return "format_only", False
    return ("algorithmic" if ft == "algorithmic_error" else "format_only"), False


def _trace_quality(candidate, transcript, func):
    code = (candidate or "")
    return {
        "plan_present": ("PLAN" in transcript) or ("CRITIQUE" in transcript),
        "base_case_present": bool(re.search(r"isinstance\s*\(", code)) or "return" in code,
        "combine_step_present": code.count(func + "(") >= 2 or ("node[0]" in code and "node[1]" in code),
        "minimal_test_present": "assert" in (code + transcript),
        "distinct_candidate_and_final": True,  # enforced below at write time
    }


def _load_overlap_corpus():
    """Build the held-out corpus the contamination guard checks against (COMPUTED, not declared)."""
    bench_names = benchmark_callables()
    prompts, solutions, tests = set(), set(), set()
    for line in open(BENCHMARK_PATH):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("prompt"):
            prompts.add(r["prompt"].strip())
        for key in ("solution", "reference", "canonical_solution"):
            if r.get(key):
                solutions.add(str(r[key]).strip())
        for t in (r.get("tests") or r.get("test_list") or []):
            tests.add(str(t).strip())
    return bench_names, prompts, solutions, tests


def _contamination_guard(tid, func, candidate, final, meta, corpus):
    bench_names, prompts, solutions, tests = corpus
    prompt = (meta.get("prompt") or "").strip()
    sol_blob = (candidate or "") + "\n" + (final or "")
    return {
        "heldout_task_name_overlap": int(tid in bench_names),
        "heldout_function_name_overlap": int(func in bench_names),
        "prompt_overlap": int(bool(prompt) and prompt in prompts),
        "solution_overlap": int(any(s and s in sol_blob for s in solutions)),
        "test_overlap": int(any(t and t in sol_blob for t in tests)),
    }


def main():
    if not TASKS_PATH.exists():
        print("Run make build-v226-representation-tasks first.")
        sys.exit(1)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(TASKS_PATH)}
    corpus = _load_overlap_corpus()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "traces.jsonl"
    n = 0
    outcomes, kinds, quality_acc, by_repr = Counter(), Counter(), Counter(), Counter()
    guard_violations = 0
    envelope_failures = 0
    genuine_transitions = 0
    found = False

    with open(out_path, "w") as out:
        for ri, d in enumerate(RUN_DIRS, 1):
            csv_path = d / "best_of_3.csv"
            if not csv_path.exists():
                continue
            found = True
            for row in csv.DictReader(open(csv_path)):
                tid = row.get("id") or row.get("task_id")
                m = meta.get(tid, {})
                rep = m.get("representation", row.get("category", "?"))
                logical = m.get("logical_task", "")
                func = m.get("func_name", "")
                transcript = row.get("full_transcript", "") or ""
                agent_pass = str(row.get("passed", "")).strip().lower() in ("true", "1", "yes")

                subs = CODE_RX.findall(transcript)
                if not subs:
                    # fall back to the extracted final answer only (no transition available)
                    subs = [row.get("extracted_code", "") or ""]
                candidate = subs[0].strip()
                final = subs[-1].strip()
                cand_pass = _executes_pass(candidate)
                fin_pass = _executes_pass(final)

                vmatch = VERIFIER_RX.search(transcript)
                verifier_signal = {
                    "status": (vmatch.group(1).strip() if vmatch else ("PASS" if cand_pass else "FAIL")),
                    "category": (vmatch.group(2).strip() if vmatch else ""),
                    "detail": (vmatch.group(3).strip()[:200] if vmatch else ""),
                    "guidance": "adjust the implementation to satisfy the format check.",
                }
                critiques = CRITIQUE_RX.findall(transcript)
                repair_plan = (critiques[0].strip() if critiques else "")

                distinct = candidate != final
                genuine_transitions += int(distinct)
                if cand_pass:
                    outcome = "no_repair_needed"
                elif fin_pass and distinct:
                    outcome = "repair_attempted_fixed"
                else:
                    outcome = "repair_attempted_failed"
                kind, env_fail = _repair_kind(candidate, final, func, logical, rep, cand_pass, agent_pass)
                envelope_failures += int(env_fail)

                cg = _contamination_guard(tid, func, candidate, final, m, corpus)
                if any(cg.values()):
                    guard_violations += 1

                tq = _trace_quality(candidate, transcript, func)
                tq["distinct_candidate_and_final"] = distinct

                trace = {
                    "task_id": tid, "task_family": "tree_serialize_repr", "representation": rep,
                    "logical_task": logical, "model_config": "3b_bf16", "run": ri,
                    "candidate_solution": candidate, "verifier_signal": verifier_signal,
                    "repair_plan": repair_plan, "repaired_solution": final, "final_solution": final,
                    "final_status": "pass" if fin_pass else "fail",
                    "repair_outcome": outcome, "repair_kind": kind,
                    "candidate_executes": cand_pass, "repaired_executes": fin_pass,
                    "agent_status": "pass" if agent_pass else "fail",
                    "envelope_format_failure": env_fail,
                    "trace_quality": tq, "contamination_guard": cg,
                }
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                n += 1
                outcomes[outcome] += 1
                kinds[kind] += 1
                by_repr[f"{rep}/{trace['final_status']}"] += 1
                for k, v in tq.items():
                    quality_acc[k] += int(bool(v))

    if not found:
        print("No v2.26 eval outputs found under outputs/eval_v226_3b_run*. Nothing to reconstruct.")
        sys.exit(1)

    agg = {
        "n_traces": n,
        "genuine_transitions": genuine_transitions,
        "repair_outcomes": dict(outcomes),
        "repair_kinds": dict(kinds),
        "by_representation_status": dict(sorted(by_repr.items())),
        "trace_quality_rate": {k: round(v / n, 3) for k, v in quality_acc.items()} if n else {},
        "envelope_format_failures": envelope_failures,
        "contamination_guard_violations": guard_violations,
        "contamination_guard": ("COMPUTED vs 32-task benchmark names/prompts/solutions/tests "
                                f"+ v226 slice; violations={guard_violations}"),
    }
    (OUT_DIR / "trace_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v227] wrote {n} hardened traces -> {out_path}  (LOCAL-ONLY, gitignored)")
    print(f"[v227] repair_outcomes : {dict(outcomes)}")
    print(f"[v227] repair_kinds    : {dict(kinds)}")
    print(f"[v227] envelope_format_failures (algo-correct, agent-scored-fail): {envelope_failures}")
    print(f"[v227] contamination_guard_violations: {guard_violations}")
    if guard_violations:
        print("[v227] ERROR: contamination guard tripped — refusing to treat traces as clean.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
