"""
scripts/build_v219b_family_records.py — v2.19b family-targeted repair memory.

v2.19 showed the retrieval bottleneck is memory COVERAGE: the 99-record pool has no
verified interval/tree/rle/dict repairs, so retrieval surfaces family-irrelevant records.
This script authors verified repair records in those four families to test whether
same-family coverage helps the 32-task benchmark.

CONTAMINATION GUARD (critical): every authored task is a DIFFERENT algorithm from the 32
benchmark tasks — different function name, different output. The script asserts the authored
function names are disjoint from the benchmark callables, and verifies each solution by
executing it in a subprocess (must print PASS, exit 0) before emitting it. This keeps the
benchmark independent: the memory provides family *technique* examples, never the answer to a
benchmark task.

Output (committed, reviewable source): data/v219b_family_repair_records.jsonl
The protected indexes (memory/index_adapted, index_adapted_v29) are never touched.

Usage:
    python scripts/build_v219b_family_records.py
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "data" / "v219b_family_repair_records.jsonl"
BENCHMARK_TASKS = ROOT / "data" / "v210_clean_repair_generalisation_tasks.jsonl"

# ── Authored family records: (func_name, family, task_prompt, code) ──────────────
# Each code block defines the solution AND its own asserts + print('PASS').
# All function names are deliberately distinct from the benchmark callables.

RECORDS: list[tuple[str, str, str, str]] = [
    # ── interval family (benchmark: interval_union/insert/min_meeting_rooms/
    #    min_removals/range_summary/interval_intersection) ──────────────────────
    ("can_attend_meetings", "interval",
     "Write `can_attend_meetings(intervals)` that returns True iff no two intervals "
     "(start, end) overlap. Verify on overlapping and non-overlapping inputs.",
     """def can_attend_meetings(intervals):
    s = sorted(intervals)
    for i in range(1, len(s)):
        if s[i][0] < s[i-1][1]:
            return False
    return True
assert can_attend_meetings([(0,30),(5,10),(15,20)]) == False
assert can_attend_meetings([(7,10),(2,4)]) == True
assert can_attend_meetings([]) == True
print('PASS')"""),

    ("min_arrows", "interval",
     "Write `min_arrows(points)` returning the minimum number of arrows to burst all "
     "balloons, where each balloon is an interval (start, end) and an arrow at x bursts "
     "every balloon with start <= x <= end. Verify on overlapping and disjoint inputs.",
     """def min_arrows(points):
    if not points:
        return 0
    pts = sorted(points, key=lambda p: p[1])
    arrows = 1
    end = pts[0][1]
    for s, e in pts[1:]:
        if s > end:
            arrows += 1
            end = e
    return arrows
assert min_arrows([(10,16),(2,8),(1,6),(7,12)]) == 2
assert min_arrows([(1,2),(3,4),(5,6),(7,8)]) == 4
assert min_arrows([]) == 0
print('PASS')"""),

    ("total_covered_length", "interval",
     "Write `total_covered_length(intervals)` returning the total length covered by the "
     "union of intervals (overlaps counted once). Verify on overlapping inputs.",
     """def total_covered_length(intervals):
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs)
assert total_covered_length([(1,3),(2,5),(7,9)]) == 6
assert total_covered_length([(0,10)]) == 10
assert total_covered_length([]) == 0
print('PASS')"""),

    ("point_coverage_count", "interval",
     "Write `point_coverage_count(intervals, point)` returning how many intervals "
     "(start, end) contain the given point (inclusive). Verify on a few inputs.",
     """def point_coverage_count(intervals, point):
    return sum(1 for s, e in intervals if s <= point <= e)
assert point_coverage_count([(1,5),(2,6),(8,10)], 3) == 2
assert point_coverage_count([(1,5)], 9) == 0
assert point_coverage_count([], 0) == 0
print('PASS')"""),

    # ── tree family (nested tuple: leaf=int, branch=(left,right)).
    #    benchmark: tree_sum/leaves/mirror/max_path/from_sorted/to_str/width ─────
    ("tree_height", "tree",
     "A tree is a nested tuple where a leaf is an int and a branch is (left, right). "
     "Write `tree_height(node)` returning the number of levels (a single leaf has "
     "height 1). Verify on nested inputs.",
     """def tree_height(node):
    if isinstance(node, tuple):
        return 1 + max(tree_height(node[0]), tree_height(node[1]))
    return 1
assert tree_height(1) == 1
assert tree_height((1,2)) == 2
assert tree_height((1,(2,3))) == 3
print('PASS')"""),

    ("tree_count_nodes", "tree",
     "A tree is a nested tuple where a leaf is an int and a branch is (left, right). "
     "Write `tree_count_nodes(node)` returning the total number of nodes (leaves and "
     "branches). Verify on nested inputs.",
     """def tree_count_nodes(node):
    if isinstance(node, tuple):
        return 1 + tree_count_nodes(node[0]) + tree_count_nodes(node[1])
    return 1
assert tree_count_nodes(1) == 1
assert tree_count_nodes((1,2)) == 3
assert tree_count_nodes((1,(2,3))) == 5
print('PASS')"""),

    ("tree_contains", "tree",
     "A tree is a nested tuple where a leaf is an int and a branch is (left, right). "
     "Write `tree_contains(node, target)` returning True iff target appears as a leaf. "
     "Verify present and absent cases.",
     """def tree_contains(node, target):
    if isinstance(node, tuple):
        return tree_contains(node[0], target) or tree_contains(node[1], target)
    return node == target
assert tree_contains((1,(2,3)), 3) == True
assert tree_contains((1,2), 5) == False
assert tree_contains(7, 7) == True
print('PASS')"""),

    ("tree_min_leaf", "tree",
     "A tree is a nested tuple where a leaf is an int and a branch is (left, right). "
     "Write `tree_min_leaf(node)` returning the smallest leaf value. Verify on nested "
     "inputs.",
     """def tree_min_leaf(node):
    if isinstance(node, tuple):
        return min(tree_min_leaf(node[0]), tree_min_leaf(node[1]))
    return node
assert tree_min_leaf((5,(2,8))) == 2
assert tree_min_leaf(7) == 7
print('PASS')"""),

    # ── rle family (pairs = list of (char, count)).
    #    benchmark: rle_roundtrip/compress/expand/longest_run/delta_encode ───────
    ("rle_total_length", "rle",
     "Run-length data is a list of (char, count) pairs. Write `rle_total_length(pairs)` "
     "returning the length of the expanded string. Verify on a few inputs.",
     """def rle_total_length(pairs):
    return sum(c for _, c in pairs)
assert rle_total_length([('a',3),('b',2)]) == 5
assert rle_total_length([]) == 0
print('PASS')"""),

    ("rle_char_at", "rle",
     "Run-length data is a list of (char, count) pairs. Write `rle_char_at(pairs, idx)` "
     "returning the character at expanded position idx (0-based). Verify across pair "
     "boundaries.",
     """def rle_char_at(pairs, idx):
    for ch, c in pairs:
        if idx < c:
            return ch
        idx -= c
    raise IndexError('index out of range')
assert rle_char_at([('a',3),('b',2)], 0) == 'a'
assert rle_char_at([('a',3),('b',2)], 3) == 'b'
assert rle_char_at([('a',3),('b',2)], 4) == 'b'
print('PASS')"""),

    ("rle_distinct_chars", "rle",
     "Run-length data is a list of (char, count) pairs. Write `rle_distinct_chars(pairs)` "
     "returning the number of distinct characters. Verify with repeated characters.",
     """def rle_distinct_chars(pairs):
    return len({ch for ch, _ in pairs})
assert rle_distinct_chars([('a',3),('b',2),('a',1)]) == 2
assert rle_distinct_chars([]) == 0
print('PASS')"""),

    ("rle_most_common", "rle",
     "Run-length data is a list of (char, count) pairs. Write `rle_most_common(pairs)` "
     "returning the character with the greatest total count. Verify with split runs.",
     """def rle_most_common(pairs):
    totals = {}
    for ch, c in pairs:
        totals[ch] = totals.get(ch, 0) + c
    return max(totals, key=lambda k: totals[k])
assert rle_most_common([('a',3),('b',2),('a',1)]) == 'a'
assert rle_most_common([('x',1),('y',5)]) == 'y'
print('PASS')"""),

    # ── dict family (nested dict).
    #    benchmark: deep_delete/flatten_dict/deep_merge/deep_keys/unflatten_dict/
    #    deep_update_if/deep_filter ──────────────────────────────────────────────
    ("deep_get", "dict",
     "Write `deep_get(d, path, default=None)` that follows a list of keys into a nested "
     "dict and returns the value, or default if any key is missing. Verify present and "
     "missing paths.",
     """def deep_get(d, path, default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur
assert deep_get({'a':{'b':{'c':1}}}, ['a','b','c']) == 1
assert deep_get({'a':1}, ['a','x'], 0) == 0
print('PASS')"""),

    ("deep_values", "dict",
     "Write `deep_values(d)` returning a flat list of all leaf values in a nested dict, "
     "in insertion order. Verify on a nested dict.",
     """def deep_values(d):
    out = []
    for v in d.values():
        if isinstance(v, dict):
            out.extend(deep_values(v))
        else:
            out.append(v)
    return out
assert deep_values({'a':1,'b':{'c':2,'d':3}}) == [1,2,3]
assert deep_values({}) == []
print('PASS')"""),

    ("deep_count_leaves", "dict",
     "Write `deep_count_leaves(d)` returning the number of leaf (non-dict) values in a "
     "nested dict. Verify on nested and empty inputs.",
     """def deep_count_leaves(d):
    n = 0
    for v in d.values():
        n += deep_count_leaves(v) if isinstance(v, dict) else 1
    return n
assert deep_count_leaves({'a':1,'b':{'c':2,'d':3}}) == 3
assert deep_count_leaves({}) == 0
print('PASS')"""),

    ("deep_max_depth", "dict",
     "Write `deep_max_depth(d)` returning the maximum nesting depth of a dict (a flat "
     "dict has depth 1, an empty dict depth 0). Verify on nested inputs.",
     """def deep_max_depth(d):
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(deep_max_depth(v) for v in d.values())
assert deep_max_depth({'a':1}) == 1
assert deep_max_depth({'a':{'b':{'c':1}}}) == 3
assert deep_max_depth({}) == 0
print('PASS')"""),
]


def benchmark_callables() -> set[str]:
    """Extract the backtick function names from the 32-task benchmark prompts."""
    names = set()
    rx = re.compile(r"`([a-zA-Z_]\w*)\s*\(")
    for line in open(BENCHMARK_TASKS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        text = r.get("prompt") or r.get("task", "")
        for m in rx.finditer(text):
            names.add(m.group(1))
        # also the task id stem
        names.add(r.get("id", "").replace("v210_", ""))
    return names


def verify(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "cand.py"
        fp.write_text(code)
        try:
            cp = subprocess.run([sys.executable, str(fp)], capture_output=True,
                                text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    ok = cp.returncode == 0 and "PASS" in (cp.stdout or "")
    return ok, (cp.stdout + cp.stderr).strip()


def main():
    bench = benchmark_callables()
    print(f"[v219b] benchmark callables to avoid: {sorted(bench)}")

    # Contamination guard: authored names must be disjoint from benchmark names.
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
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {family:9} {func}")
        if not ok:
            print(f"      output: {out[:300]}")
            print("ERROR: authored record failed verification; aborting.", file=sys.stderr)
            sys.exit(1)
        ctc = "TOOL_CALL: execute_code(" + json.dumps({"code": code}) + ")"
        records.append({
            "id": str(uuid.uuid4()),
            "task": prompt,
            "category": "family_repair",
            "failure_type": "none",
            "query_text": f"{family} {prompt}",
            "corrected_tool_call": ctc,
            "observation": "PASS",
            "final_answer": "",
            "source": "v219b_family_authored",
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
    print(f"\n[v219b] verified {len(records)} family records -> {OUT_PATH}")
    print(f"[v219b] families: {dict(fam)}")
    print(f"[v219b] contamination guard PASSED (0 overlap with {len(bench)} benchmark names)")


if __name__ == "__main__":
    main()
