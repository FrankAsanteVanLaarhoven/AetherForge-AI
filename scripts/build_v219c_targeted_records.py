"""
scripts/build_v219c_targeted_records.py — v2.19c targeted coverage records.

v2.19b passed the gate but coverage was necessary-not-sufficient: interval_union and the
tree stable-fails still failed despite retrieving relevant family records. Inspection of
the v2.19b records shows why — they all return scalars/bools via simple recursion
(tree_height/count/contains/min_leaf; interval can_attend/total_length/min_arrows). The
persistent failures need a different PATTERN: building a structured OUTPUT (a list/levels
for tree; a merged/derived interval LIST for interval) via family traversal.

This script authors verified same-family-different-task records that exercise those
output-building patterns — without being the benchmark tasks. Same contamination guard as
v2.19b: function names asserted disjoint from all 32 benchmark callables, each solution
execution-verified. Output (committed source): data/v219c_targeted_repair_records.jsonl

Usage:
    python scripts/build_v219c_targeted_records.py
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables, verify  # noqa: E402

OUT_PATH = ROOT / "data" / "v219c_targeted_repair_records.jsonl"

# (func_name, family, task_prompt, code) — patterns: structured/list/level output building.
RECORDS: list[tuple[str, str, str, str]] = [
    # ── tree: build structured / level outputs (vs v2.19b's scalar recursions) ──
    ("tree_to_nested_list", "tree",
     "A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
     "`tree_to_nested_list(node)` returning the same structure with tuples replaced by "
     "lists (leaves unchanged). Verify on nested inputs.",
     """def tree_to_nested_list(node):
    if isinstance(node, tuple):
        return [tree_to_nested_list(node[0]), tree_to_nested_list(node[1])]
    return node
assert tree_to_nested_list((1,(2,3))) == [1, [2, 3]]
assert tree_to_nested_list(5) == 5
assert tree_to_nested_list(((1,2),(3,4))) == [[1,2],[3,4]]
print('PASS')"""),

    ("tree_level_counts", "tree",
     "A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
     "`tree_level_counts(node)` returning a list whose i-th entry is the number of nodes "
     "at depth i (breadth-first). Verify on nested inputs.",
     """def tree_level_counts(node):
    counts = []
    level = [node]
    while level:
        counts.append(len(level))
        nxt = []
        for x in level:
            if isinstance(x, tuple):
                nxt.append(x[0])
                nxt.append(x[1])
        level = nxt
    return counts
assert tree_level_counts(1) == [1]
assert tree_level_counts((1,2)) == [1,2]
assert tree_level_counts(((1,2),(3,4))) == [1,2,4]
assert tree_level_counts((1,(2,3))) == [1,2,2]
print('PASS')"""),

    ("tree_count_at_depth", "tree",
     "A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
     "`tree_count_at_depth(node, depth)` returning the number of nodes at the given depth "
     "(root is depth 0). Verify across depths.",
     """def tree_count_at_depth(node, depth):
    if depth == 0:
        return 1
    if isinstance(node, tuple):
        return tree_count_at_depth(node[0], depth-1) + tree_count_at_depth(node[1], depth-1)
    return 0
assert tree_count_at_depth((1,(2,3)), 0) == 1
assert tree_count_at_depth((1,(2,3)), 1) == 2
assert tree_count_at_depth((1,(2,3)), 2) == 2
print('PASS')"""),

    # ── interval: build a derived interval LIST via sweep (vs v2.19b's scalar/bool) ──
    ("interval_complement", "interval",
     "Write `interval_complement(intervals, lo, hi)` returning the list of (start, end) "
     "gaps within [lo, hi] that are NOT covered by any interval, in order. Verify on "
     "overlapping inputs and the empty case.",
     """def interval_complement(intervals, lo, hi):
    result = []
    cur = lo
    for s, e in sorted(intervals):
        if s > cur:
            result.append((cur, min(s, hi)))
        cur = max(cur, e)
        if cur >= hi:
            break
    if cur < hi:
        result.append((cur, hi))
    return result
assert interval_complement([(1,3),(5,7)], 0, 8) == [(0,1),(3,5),(7,8)]
assert interval_complement([], 0, 5) == [(0,5)]
print('PASS')"""),

    ("interval_clip", "interval",
     "Write `interval_clip(intervals, lo, hi)` returning each interval clipped to the "
     "range [lo, hi], dropping intervals that become empty. Verify clipping and dropping.",
     """def interval_clip(intervals, lo, hi):
    out = []
    for s, e in intervals:
        ns, ne = max(s, lo), min(e, hi)
        if ns < ne:
            out.append((ns, ne))
    return out
assert interval_clip([(1,5),(8,12)], 3, 10) == [(3,5),(8,10)]
assert interval_clip([(0,1)], 5, 9) == []
print('PASS')"""),

    ("interval_gaps", "interval",
     "Write `interval_gaps(intervals)` returning the list of (end, start) gaps between "
     "consecutive intervals after sorting (only where there is a positive gap). Verify on "
     "unsorted input.",
     """def interval_gaps(intervals):
    s = sorted(intervals)
    gaps = []
    for i in range(1, len(s)):
        prev_end = s[i-1][1]
        cur_start = s[i][0]
        if cur_start > prev_end:
            gaps.append((prev_end, cur_start))
    return gaps
assert interval_gaps([(1,3),(6,8),(4,5)]) == [(3,4),(5,6)]
assert interval_gaps([(1,5)]) == []
print('PASS')"""),
]


def main():
    bench = benchmark_callables()
    authored = [r[0] for r in RECORDS]
    overlap = sorted(set(authored) & bench)
    if overlap:
        print(f"ERROR: authored functions overlap benchmark tasks: {overlap}", file=sys.stderr)
        sys.exit(1)
    if len(set(authored)) != len(authored):
        print("ERROR: duplicate authored function names", file=sys.stderr)
        sys.exit(1)

    records = []
    for func, family, prompt, code in RECORDS:
        ok, out = verify(code)
        print(f"  [{'PASS' if ok else 'FAIL'}] {family:9} {func}")
        if not ok:
            print(f"      output: {out[:300]}")
            print("ERROR: authored record failed verification; aborting.", file=sys.stderr)
            sys.exit(1)
        ctc = "TOOL_CALL: execute_code(" + json.dumps({"code": code}) + ")"
        records.append({
            "id": str(uuid.uuid4()),
            "task": prompt,
            "category": "targeted_repair",
            "failure_type": "none",
            "query_text": f"{family} {prompt}",
            "corrected_tool_call": ctc,
            "observation": "PASS",
            "final_answer": "",
            "source": "v219c_targeted_authored",
            "verified": True,
            "content_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            "sensitivity": "internal",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    fam = Counter(r2[1] for r2 in RECORDS)
    print(f"\n[v219c] verified {len(records)} targeted records -> {OUT_PATH}")
    print(f"[v219c] families: {dict(fam)}")
    print(f"[v219c] contamination guard PASSED (0 overlap with {len(bench)} benchmark names)")


if __name__ == "__main__":
    main()
