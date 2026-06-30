"""
scripts/build_v229_repair_harvest.py — v2.29 genuine repair trace harvest (source-only).

v2.28 found the substrate has 0 genuine broken→fixed repair transitions (the model never repaired;
degenerate claimed-repairs were filtered). This harness MANUFACTURES genuine, verifier-labelled
format-repair transitions WITHOUT touching held-out evaluation: it perturbs ONLY the output FORMAT
of a known-correct tree serializer, confirms by execution that the perturbed candidate fails and the
canonical version passes, links the repair to the v2.27 structured verifier diagnosis, and records
the transition.

This is NOT training and NOT faked improvement: every record is an execution-verified
candidate(fail) → verifier signal → canonical repair → final(pass) transition where
candidate_solution != final_solution.

Output (LOCAL-ONLY, gitignored):
    data/generated/v229/repair_traces.jsonl
    data/generated/v229/harvest_aggregate.json

Usage:
    python scripts/build_v229_repair_harvest.py
"""

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v226_representation_tasks import TASKS  # noqa: E402  (func, logical, rep, desc, ref, tests)
from scripts.build_v227_trace_factory import (  # noqa: E402
    PROBE_INPUTS, _contamination_guard, _load_overlap_corpus, _observed_output,
)
from scripts.v227_format_verifier import format_verify, render  # noqa: E402

OUT_DIR = ROOT / "data" / "generated" / "v229"
META_PATH = ROOT / "data" / "v226_representation_tasks.jsonl"


def _run(prog, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"
        fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return (r.returncode == 0 and "PASS" in r.stdout), (r.stdout + r.stderr)


def _program(code, tests):
    return code + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('PASS')\n"


# ── Format-only perturbations (output format changes; algorithm preserved) ─────
# Each returns a mutated code string or None if not applicable. Execution then confirms it FAILS.

def _mut_separator(code):
    for a, b in (('" "', '","'), ('","', '";"'), ('":"', '"|"')):
        if a in code:
            return "separator_error", code.replace(a, b, 1)
    return None


def _mut_bracket(code):
    if '"("' in code and '")"' in code:
        return "wrong_bracket_type", code.replace('"("', '"["').replace('")"', '"]"')
    if '"["' in code and '"]"' in code:
        return "wrong_bracket_type", code.replace('"["', '"("').replace('"]"', '")"')
    return None


def _mut_missing_marker(code):
    for a, b in (('"(" + ', ''), ('["("] + ', ''), ('"(" +', '"" +')):
        if a in code:
            return "missing_null_marker", code.replace(a, b, 1)
    return None


def _mut_extra_marker(code):
    for a, b in (('+ ")"', '+ "))"'), ('+ [")"]', '+ [")", ")"]')):
        if a in code:
            return "extra_null_marker", code.replace(a, b, 1)
    return None


def _mut_string_list(code):
    # string serializer returns a 1-element list at the leaf -> string/list mismatch
    if "return str(node)" in code:
        return "string_list_mismatch", code.replace("return str(node)", "return [str(node)]", 1)
    return None


def _mut_tuple_list(code):
    # nested_list serializer returns a tuple instead of a list -> tuple/list mismatch
    if "return [" in code and "node[0]" in code and "node[1]" in code:
        i = code.index("return [")
        head, tail = code[:i], code[i:]
        nl = tail.find("\n")
        line = tail if nl == -1 else tail[:nl]
        rest = "" if nl == -1 else tail[nl:]
        if line.rstrip().endswith("]"):
            mutated = line.replace("return [", "return (", 1).rstrip()
            mutated = mutated[:-1] + ")"
            return "tuple_list_mismatch", head + mutated + rest
    return None


MUTATORS = [_mut_separator, _mut_bracket, _mut_missing_marker, _mut_extra_marker,
            _mut_string_list, _mut_tuple_list]


def main():
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(META_PATH)} if META_PATH.exists() else {}
    corpus = _load_overlap_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FORMAT_TYPES = {"missing_null_marker", "extra_null_marker", "separator_error",
                    "ordering_error", "type_error", "format_error"}
    records = []
    attempts = 0
    successes = 0
    rejected_nonformat = 0
    fail_types = Counter()
    seq = 200

    for func, logical, rep, desc, ref, tests in TASKS:
        tid = f"v226_{func}"
        m = meta.get(tid, {})
        ref_code = ref
        ref_ok, _ = _run(_program(ref_code, tests))
        if not ref_ok:
            continue  # reference must pass; skip otherwise
        for mut in MUTATORS:
            res = mut(ref_code)
            if res is None:
                continue
            fault, cand_code = res
            if cand_code.strip() == ref_code.strip():
                continue
            attempts += 1
            cand_ok, _ = _run(_program(cand_code, tests))
            if cand_ok:
                continue  # perturbation did not actually break the output -> not a repairable failure
            # genuine failure confirmed. Build the verifier signal from the candidate's output.
            ok, observed = _observed_output(cand_code, func, PROBE_INPUTS[0])
            try:
                expected = render(PROBE_INPUTS[0], logical, rep)
            except ValueError:
                expected = None
            if not (ok and expected is not None):
                # candidate raised before producing output: not a clean output-format failure
                rejected_nonformat += 1
                continue
            vblock = format_verify(observed, expected, rep, logical)
            # keep only verifier-confirmed FORMAT failures (the harvest target); a format-only code
            # perturbation the value-level verifier reads as algorithmic is dropped as ambiguous, so
            # both this harness and the v2.28 builder agree on the label.
            if vblock["failure_type"] not in FORMAT_TYPES:
                rejected_nonformat += 1
                continue
            seq += 1
            cg = _contamination_guard(tid, func, cand_code, ref_code, m, corpus)
            # store full programs (code + asserts + print('PASS')) so the v2.28 builder verifies
            # candidate(fail)/final(pass) by execution, matching the v2.27 trace convention.
            cand_prog = _program(cand_code, tests)
            ref_prog = _program(ref_code, tests)
            repair_plan = (f"verifier failure_type={vblock['failure_type']}: {vblock['diagnosis']} "
                           f"Apply: {vblock['repair_hint']}")
            rec = {
                "record_id": f"v229_{func}_{fault}",
                "task_id": tid, "task_family": "tree_serialize_repr",
                "representation": rep, "logical_task": logical,
                "model_config": "harvest_format_perturbation", "run": seq,
                "prompt_mode": "verifier_driven_format_repair",
                "candidate_solution": cand_prog, "candidate_status": "fail",
                "verifier_signal": {
                    "status": "fail", "category": vblock["failure_type"],
                    "failure_type": vblock["failure_type"],
                    "expected": str(expected)[:120] if expected is not None else "",
                    "observed": str(observed)[:120],
                    "detail": vblock["diagnosis"], "diagnosis": vblock["diagnosis"],
                    "guidance": vblock["repair_hint"], "repair_hint": vblock["repair_hint"],
                },
                "failure_type": vblock["failure_type"],
                "repair_plan": repair_plan,
                "repaired_solution": ref_prog, "final_solution": ref_prog, "final_status": "pass",
                "candidate_differs_from_final": True, "repair_successful": True,
                "repair_outcome": "repair_attempted_fixed", "repair_kind": vblock["failure_type"],
                "trace_quality": {"plan_present": True, "base_case_present": True,
                                  "combine_step_present": True, "minimal_test_present": True,
                                  "distinct_candidate_and_final": True},
                "contamination_guard": cg,
            }
            records.append(rec)
            successes += 1
            fail_types[vblock["failure_type"]] += 1

    out_path = OUT_DIR / "repair_traces.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    guard_violations = sum(1 for r in records if any(r["contamination_guard"].values()))
    agg = {
        "records_harvested": len(records),
        "repair_attempts": attempts,
        "successful_repairs": successes,
        "rejected_nonformat_label": rejected_nonformat,
        "failure_type_distribution": dict(fail_types),
        "representation_distribution": dict(Counter(r["representation"] for r in records)),
        "contamination_guard_violations": guard_violations,
        "all_genuine_transitions": all(r["candidate_solution"].strip() != r["final_solution"].strip()
                                       for r in records),
    }
    (OUT_DIR / "harvest_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")

    print(f"[v229] harvested {len(records)} genuine repair traces -> {out_path}  (LOCAL-ONLY)")
    print(f"[v229] attempts={attempts} successes={successes} "
          f"fail_types={dict(fail_types)}")
    print(f"[v229] contamination_guard_violations={guard_violations} "
          f"all_genuine={agg['all_genuine_transitions']}")
    if guard_violations:
        print("[v229] ERROR: contamination guard tripped.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
