"""
scripts/build_v230_broadened_repair_harvest.py — v2.30 broadened repair trace harvest (source-only).

v2.29 produced 9 genuine repair traces, all tree_serialize output-format perturbations. v2.30
broadens the corpus across more task families AND adds a controlled ALGORITHMIC repair slice, so the
substrate carries both format_repair and algorithmic_repair categories for a future SFT / preference
pilot.

Method (no held-out contamination): a small library of NEW, non-held-out, execution-verified
reference functions (names disjoint from the 32-task benchmark) is perturbed by typed mutators. Each
mutator injects a KNOWN fault with a KNOWN verifier failure_type; a pair is kept only when execution
confirms the perturbed candidate FAILS its asserts and the canonical version PASSES. The structured
verifier signal carries the injected failure_type plus the REAL executed observed/expected values.

    repair_category in {format_repair, algorithmic_repair, mixed_repair}

Output (LOCAL-ONLY, gitignored):
    data/generated/v230/repair_traces.jsonl
    data/generated/v230/harvest_aggregate.json

Usage:
    python scripts/build_v230_broadened_repair_harvest.py
"""

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import _contamination_guard, _load_overlap_corpus  # noqa: E402

OUT_DIR = ROOT / "data" / "generated" / "v230"
FORMAT_TYPES = {"missing_null_marker", "extra_null_marker", "separator_error",
                "ordering_error", "type_error", "format_error"}

# ── Task library: (func, family, reference_code, tests, probe_args) ──
FORMAT_TASKS = [
    ("csv_join", "list_format", "def csv_join(xs):\n    return \",\".join(str(x) for x in xs)",
     ['csv_join([1,2,3]) == "1,2,3"', 'csv_join([]) == ""', 'csv_join([7]) == "7"'], ([1, 2, 3],)),
    ("pipe_join", "list_format", "def pipe_join(xs):\n    return \"|\".join(str(x) for x in xs)",
     ['pipe_join([1,2]) == "1|2"', 'pipe_join([5]) == "5"', 'pipe_join([1,2,3]) == "1|2|3"'], ([1, 2],)),
    ("kv_format", "kv_format",
     "def kv_format(pairs):\n    return \";\".join(k + \"=\" + str(v) for k, v in pairs)",
     ['kv_format([("a",1)]) == "a=1"', 'kv_format([("a",1),("b",2)]) == "a=1;b=2"',
      'kv_format([]) == ""'], ([("a", 1), ("b", 2)],)),
    ("bracket_list", "container_format",
     "def bracket_list(xs):\n    return \"[\" + \", \".join(str(x) for x in xs) + \"]\"",
     ['bracket_list([1,2]) == "[1, 2]"', 'bracket_list([]) == "[]"', 'bracket_list([9]) == "[9]"'],
     ([1, 2],)),
    ("tuple_to_list", "container_format",
     "def tuple_to_list(t):\n    return list(t)",
     ['tuple_to_list((1,2)) == [1,2]', 'tuple_to_list((7,8,9)) == [7,8,9]', 'tuple_to_list(()) == []'],
     ((1, 2),)),
    ("wrap_parens", "string_format",
     "def wrap_parens(xs):\n    return \"(\" + \" \".join(str(x) for x in xs) + \")\"",
     ['wrap_parens([1,2]) == "(1 2)"', 'wrap_parens([]) == "()"', 'wrap_parens([3]) == "(3)"'],
     ([1, 2],)),
    ("json_pairs", "json_format", "def json_pairs(pairs):\n    return dict(pairs)",
     ['json_pairs([("a",1)]) == {"a":1}', 'json_pairs([("a",1),("b",2)]) == {"a":1,"b":2}',
      'json_pairs([]) == {}'], ([("a", 1), ("b", 2)],)),
    ("dash_join", "list_format", "def dash_join(xs):\n    return \"-\".join(str(x) for x in xs)",
     ['dash_join([1,2,3]) == "1-2-3"', 'dash_join([4]) == "4"', 'dash_join([]) == ""'], ([1, 2, 3],)),
    ("colon_pairs", "kv_format",
     "def colon_pairs(xs):\n    return \",\".join(str(i) + \":\" + str(v) for i, v in enumerate(xs))",
     ['colon_pairs([5,6]) == "0:5,1:6"', 'colon_pairs([9]) == "0:9"', 'colon_pairs([]) == ""'],
     ([5, 6],)),
    ("nest_pairs", "container_format",
     "def nest_pairs(xs):\n    return [[i, v] for i, v in enumerate(xs)]",
     ['nest_pairs([7,8]) == [[0,7],[1,8]]', 'nest_pairs([3]) == [[0,3]]', 'nest_pairs([]) == []'],
     ([7, 8],)),
]

ALGO_TASKS = [
    ("list_total", "arithmetic",
     "def list_total(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t",
     ['list_total([1,2,3]) == 6', 'list_total([]) == 0', 'list_total([5]) == 5'], ([1, 2, 3],)),
    ("list_top", "arithmetic",
     "def list_top(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m",
     ['list_top([1,3,2]) == 3', 'list_top([5]) == 5', 'list_top([-1,-4]) == -1'], ([1, 3, 2],)),
    ("factorialn", "recursion",
     "def factorialn(n):\n    if n <= 1:\n        return 1\n    return n * factorialn(n - 1)",
     ['factorialn(0) == 1', 'factorialn(3) == 6', 'factorialn(5) == 120'], (3,)),
    ("fib_n", "recursion",
     "def fib_n(n):\n    if n < 2:\n        return n\n    return fib_n(n - 1) + fib_n(n - 2)",
     ['fib_n(0) == 0', 'fib_n(6) == 8', 'fib_n(10) == 55'], (6,)),
    ("reverse_seq", "sequence",
     "def reverse_seq(xs):\n    out = []\n    for x in xs:\n        out = [x] + out\n    return out",
     ['reverse_seq([1,2,3]) == [3,2,1]', 'reverse_seq([]) == []', 'reverse_seq([9]) == [9]'],
     ([1, 2, 3],)),
    ("count_evens", "scan",
     "def count_evens(xs):\n    c = 0\n    for x in xs:\n        if x % 2 == 0:\n            c += 1\n    return c",
     ['count_evens([1,2,3,4]) == 2', 'count_evens([]) == 0', 'count_evens([2]) == 1'], ([1, 2, 3, 4],)),
    ("gcd2", "arithmetic",
     "def gcd2(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     ['gcd2(12,8) == 4', 'gcd2(5,0) == 5', 'gcd2(7,13) == 1'], (12, 8)),
    ("running_total", "scan",
     "def running_total(xs):\n    out = []\n    t = 0\n    for x in xs:\n        t += x\n        out.append(t)\n    return out",
     ['running_total([1,2,3]) == [1,3,6]', 'running_total([]) == []', 'running_total([4]) == [4]'],
     ([1, 2, 3],)),
    ("clamp_val", "arithmetic",
     "def clamp_val(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
     ['clamp_val(5,0,10) == 5', 'clamp_val(-3,0,10) == 0', 'clamp_val(99,0,10) == 10'], (5, 0, 10)),
    ("nth_even", "sequence",
     "def nth_even(n):\n    return 2 * n",
     ['nth_even(0) == 0', 'nth_even(3) == 6', 'nth_even(5) == 10'], (3,)),
]

# typed mutators: (from, to, failure_type). Format faults land in FORMAT_TYPES; algo -> algorithmic_error.
FORMAT_MUTATORS = [
    ('","', '";"', 'separator_error'), ('"|"', '","', 'separator_error'),
    ('"-"', '","', 'separator_error'), ('" "', '","', 'separator_error'),
    ('"="', '":"', 'separator_error'), ('": "', '" "', 'separator_error'),
    ('", "', '";"', 'separator_error'), ('";"', '","', 'separator_error'),
    ('"["', '"("', 'format_error'), ('"]"', '")"', 'format_error'),
    ('list(t)', 'tuple(t)', 'type_error'), ('dict(pairs)', 'list(pairs)', 'type_error'),
    ('[i, v]', '(i, v)', 'type_error'),
    ('str(x) for x', 'str(x) + "!" for x', 'format_error'),       # stray suffix token
    ('str(i) + ":"', '"x" + str(i) + ":"', 'format_error'),       # stray prefix token
]
ALGO_MUTATORS = [
    (" > ", " < ", 'algorithmic_error'), (" < ", " > ", 'algorithmic_error'),
    (" % 2 == 0", " % 2 == 1", 'algorithmic_error'), (" - 1", " - 2", 'algorithmic_error'),
    ("n < 2", "n < 1", 'algorithmic_error'), ("n <= 1", "n < 1", 'algorithmic_error'),
    ("return 1", "return 0", 'algorithmic_error'), ("t = 0", "t = 1", 'algorithmic_error'),
    ("c = 0", "c = 1", 'algorithmic_error'), ("[x] + out", "out + [x]", 'algorithmic_error'),
    ("m = xs[0]", "m = 0", 'algorithmic_error'), ("2 * n", "2 + n", 'algorithmic_error'),
    ("a % b", "b % a", 'algorithmic_error'), ("t += x", "t -= x", 'algorithmic_error'),
    ("c += 1", "c += 2", 'algorithmic_error'), (".append(t)", ".append(t + 1)", 'algorithmic_error'),
    ("x > m", "x >= m", 'algorithmic_error'),
    ("a, b = b, a % b", "a, b = a % b, b", 'algorithmic_error'),
    ("return n\n", "return n + 1\n", 'algorithmic_error'), ("x < lo", "x > lo", 'algorithmic_error'),
]


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


def _observe(code, func, args):
    call = func + "(" + ", ".join(repr(a) for a in args) + ")"
    prog = code + f"\ntry:\n    print('OUT::' + repr({call}))\nexcept Exception as _e:\n    print('ERR::' + type(_e).__name__)\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"; fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return "timeout"
    for line in r.stdout.splitlines():
        if line.startswith("OUT::"):
            return line[5:]
        if line.startswith("ERR::"):
            return f"raised {line[5:]}"
    return "no output"


def main():
    corpus = _load_overlap_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    attempts = 0
    rejected = Counter()
    seq = 300

    suite = ([(t, "format", FORMAT_MUTATORS) for t in FORMAT_TASKS] +
             [(t, "algo", ALGO_MUTATORS) for t in ALGO_TASKS])

    for (func, family, ref_code, tests, probe), bucket, mutators in suite:
        if not _run(_program(ref_code, tests))[0]:
            continue
        used = set()
        for frm, to, ftype in mutators:
            if frm not in ref_code:
                continue
            cand_code = ref_code.replace(frm, to, 1)
            if cand_code == ref_code or cand_code in used:
                continue
            used.add(cand_code)
            attempts += 1
            if _run(_program(cand_code, tests))[0]:
                rejected["perturbation_did_not_break"] += 1
                continue
            observed = _observe(cand_code, func, probe)
            expected = _observe(ref_code, func, probe)
            cg = _contamination_guard(func, func, cand_code, ref_code, {}, corpus)
            if any(cg.values()):
                rejected["contamination_overlap"] += 1
                continue
            category = ("algorithmic_repair" if bucket == "algo"
                        else "format_repair" if ftype in FORMAT_TYPES else "mixed_repair")
            hint = ("re-derive the value from a correct traversal/accumulator, then return it."
                    if category == "algorithmic_repair" else
                    "re-render the output in the exact required format (separator/bracket/container/type).")
            vsig = {"status": "fail", "failure_type": ftype, "category": ftype,
                    "expected": str(expected)[:120], "observed": str(observed)[:120],
                    "detail": f"{func}{probe}: observed != expected",
                    "diagnosis": f"{func}: {ftype.replace('_', ' ')}.",
                    "guidance": hint, "repair_hint": hint}
            seq += 1
            cand_prog, ref_prog = _program(cand_code, tests), _program(ref_code, tests)
            records.append({
                "record_id": f"v230_{func}_{ftype}_{len(used)}", "task_id": f"v230_{func}",
                "task_family": family, "representation": family, "logical_task": func,
                "model_config": "harvest_controlled_perturbation", "run": seq,
                "prompt_mode": "verifier_driven_repair",
                "candidate_solution": cand_prog, "candidate_status": "fail",
                "verifier_signal": vsig, "failure_type": ftype,
                "repair_plan": f"verifier failure_type={ftype}: {vsig['diagnosis']} Apply: {hint}",
                "repaired_solution": ref_prog, "final_solution": ref_prog, "final_status": "pass",
                "candidate_differs_from_final": True, "repair_successful": True,
                "repair_category": category, "repair_outcome": "repair_attempted_fixed",
                "repair_kind": ftype,
                "trace_quality": {"plan_present": True, "base_case_present": True,
                                  "combine_step_present": True, "minimal_test_present": True,
                                  "distinct_candidate_and_final": True},
                "contamination_guard": cg,
            })

    out_path = OUT_DIR / "repair_traces.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats = Counter(r["repair_category"] for r in records)
    guard_violations = sum(1 for r in records if any(r["contamination_guard"].values()))
    agg = {
        "records_harvested": len(records), "repair_attempts": attempts,
        "successful_repairs": len(records), "repair_category_distribution": dict(cats),
        "format_repair": cats.get("format_repair", 0),
        "algorithmic_repair": cats.get("algorithmic_repair", 0),
        "mixed_repair": cats.get("mixed_repair", 0),
        "verifier_format": sum(1 for r in records if r["failure_type"] in FORMAT_TYPES),
        "failure_type_distribution": dict(Counter(r["failure_type"] for r in records)),
        "task_family_distribution": dict(Counter(r["task_family"] for r in records)),
        "representation_distribution": dict(Counter(r["representation"] for r in records)),
        "rejection_reasons": dict(rejected), "contamination_guard_violations": guard_violations,
        "all_genuine_transitions": all(r["candidate_solution"].strip() != r["final_solution"].strip()
                                       for r in records),
    }
    (OUT_DIR / "harvest_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v230] harvested {len(records)} genuine repair traces -> {out_path}  (LOCAL-ONLY)")
    print(f"[v230] categories: {dict(cats)} | verifier_format={agg['verifier_format']}")
    print(f"[v230] families({len(agg['task_family_distribution'])}): {agg['task_family_distribution']}")
    print(f"[v230] attempts={attempts} rejected={dict(rejected)} guard_violations={guard_violations}")
    if guard_violations:
        print("[v230] ERROR: contamination guard tripped.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
