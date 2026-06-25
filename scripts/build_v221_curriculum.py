"""
scripts/build_v221_curriculum.py — v2.21 ForgeReasoningCore reasoning curriculum.

v2.19c established that tree-family stable-fails are reasoning/control-bound: the right
same-family memory is retrieved but the model still writes incorrect recursive control.
This curriculum documents, for a few NON-benchmark tree tasks, the explicit reasoning plan
and tool-action sequence that turns a retrieved pattern into correct executed code — the
structure the v2.21 EXECUTION_PLAN_SYSTEM prompt forces at inference time.

Records are committed evidence (data/v221_reasoning_curriculum.jsonl), not added to any
memory index. Function names are contamination-guarded (disjoint from the 32 benchmark
callables) and every verified_solution is execution-checked.

Usage:
    python scripts/build_v221_curriculum.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables, verify  # noqa: E402

OUT_PATH = ROOT / "data" / "v221_reasoning_curriculum.jsonl"

PLAN = ["state the task family", "identify the recursive/iterative structure",
        "define the base case", "combine child/sub results",
        "write minimal asserts", "run, inspect failure, repair once"]
ACTIONS = ["emit_plan", "draft_solution", "run_minimal_test", "inspect_failure",
           "revise_solution", "run_test_again", "final_answer"]

# (task_name, retrieved_memory_summary, base_case, combine, code) — tree family, non-benchmark.
RECORDS = [
    ("tree_height",
     "recursive aggregation over (left, right); leaf is the base case",
     "a non-tuple node has height 1",
     "1 + max(height(left), height(right))",
     """def tree_height(node):
    if isinstance(node, tuple):
        return 1 + max(tree_height(node[0]), tree_height(node[1]))
    return 1
assert tree_height(1) == 1
assert tree_height((1,(2,3))) == 3
print('PASS')"""),

    ("tree_count_at_depth",
     "recurse with a decreasing depth counter; sum subtree counts",
     "depth 0 counts the current node; a leaf above depth 0 counts 0",
     "count_at_depth(left, d-1) + count_at_depth(right, d-1)",
     """def tree_count_at_depth(node, d):
    if d == 0:
        return 1
    if isinstance(node, tuple):
        return tree_count_at_depth(node[0], d-1) + tree_count_at_depth(node[1], d-1)
    return 0
assert tree_count_at_depth((1,(2,3)), 2) == 2
print('PASS')"""),

    ("tree_to_nested_list",
     "structural rebuild by recursion; leaves pass through unchanged",
     "a non-tuple node returns itself",
     "wrap recursive results for left and right in a list",
     """def tree_to_nested_list(node):
    if isinstance(node, tuple):
        return [tree_to_nested_list(node[0]), tree_to_nested_list(node[1])]
    return node
assert tree_to_nested_list((1,(2,3))) == [1, [2, 3]]
print('PASS')"""),

    ("tree_level_counts",
     "breadth-first level traversal building a per-level count list",
     "an empty frontier ends the loop",
     "append len(level), then expand each branch into the next level",
     """def tree_level_counts(node):
    counts = []
    level = [node]
    while level:
        counts.append(len(level))
        nxt = []
        for x in level:
            if isinstance(x, tuple):
                nxt.append(x[0]); nxt.append(x[1])
        level = nxt
    return counts
assert tree_level_counts(((1,2),(3,4))) == [1,2,4]
print('PASS')"""),
]


def main():
    bench = benchmark_callables()
    names = [r[0] for r in RECORDS]
    overlap = sorted(set(names) & bench)
    if overlap:
        print(f"ERROR: curriculum names overlap benchmark: {overlap}", file=sys.stderr)
        sys.exit(1)

    records = []
    for name, mem, base, combine, code in RECORDS:
        ok, out = verify(code)
        print(f"  [{'PASS' if ok else 'FAIL'}] tree {name}")
        if not ok:
            print(f"      {out[:200]}")
            sys.exit(1)
        minimal = "\n".join(l for l in code.splitlines() if l.strip().startswith("assert"))
        records.append({
            "task_family": "tree",
            "task_name": name,
            "retrieved_memory_summary": mem,
            "reasoning_plan": PLAN,
            "tool_action_sequence": ACTIONS,
            "base_case": base,
            "combine": combine,
            "verified_solution": code,
            "minimal_tests": minimal,
            "failure_mode_prevented":
                "retrieved correct same-family memory but failed to execute correct "
                "recursive control (missing base case or wrong combine step)",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[v221] verified {len(records)} curriculum records -> {OUT_PATH}")
    print(f"[v221] contamination guard PASSED (0 overlap with {len(bench)} benchmark names)")


if __name__ == "__main__":
    main()
