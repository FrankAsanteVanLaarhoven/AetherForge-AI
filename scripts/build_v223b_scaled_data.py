"""
scripts/build_v223b_scaled_data.py — v2.23b scaled tree-capability training data.

Scales the v2.23 pilot (~49 records, 15 tasks) by adding ~12 more distinct same-family-
different-task tree tasks, to test whether the residual hard tasks are DATA-LIMITED or at a
capability ceiling. Same discipline: every task is a DIFFERENT algorithm from the benchmark,
execution-verified, name-disjoint, on the same nested-tuple representation; the hard tasks stay
evaluation-only.

Reuses the verified base TASKS from build_v223_tree_capability_data and the same record builder
(repair-trajectory {instruction, response}). Output: data/v223b_tree_capability_scaled.jsonl

Usage:
    python scripts/build_v223b_scaled_data.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402
from scripts.build_v223_tree_capability_data import TASKS as BASE_TASKS, _verify  # noqa: E402

OUT_PATH = ROOT / "data" / "v223b_tree_capability_scaled.jsonl"

EXTRA_TASKS = [
    dict(name="tree_to_indexed", pattern="serialization",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_to_indexed(node)` returning a list of (path, value) for each leaf, where "
              "path is the string of 'L'/'R' moves from the root.",
         wrong="def tree_to_indexed(node, path=\"\"):\n    if isinstance(node, tuple):\n        return tree_to_indexed(node[0], path) + tree_to_indexed(node[1], path)\n    return [(path, node)]",
         correct="def tree_to_indexed(node, path=\"\"):\n    if isinstance(node, tuple):\n        return tree_to_indexed(node[0], path + \"L\") + tree_to_indexed(node[1], path + \"R\")\n    return [(path, node)]",
         critique="the path was not extended on recursion, so every leaf reports the empty path; append 'L'/'R'.",
         tests=['tree_to_indexed((1,2)) == [("L",1),("R",2)]',
                'tree_to_indexed(5) == [("",5)]',
                'tree_to_indexed((1,(2,3))) == [("L",1),("RL",2),("RR",3)]',
                'tree_to_indexed(((1,2),3)) == [("LL",1),("LR",2),("R",3)]',
                'tree_to_indexed((1,(2,(3,4)))) == [("L",1),("RL",2),("RRL",3),("RRR",4)]']),

    dict(name="tree_to_depth_string", pattern="serialization",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_to_depth_string(node)` returning a space-separated string of the depth of "
              "each leaf, left to right (root depth 0).",
         wrong="def tree_to_depth_string(node, d=0):\n    if isinstance(node, tuple):\n        return tree_to_depth_string(node[0], d) + \" \" + tree_to_depth_string(node[1], d)\n    return str(d)",
         correct="def tree_to_depth_string(node, d=0):\n    if isinstance(node, tuple):\n        return tree_to_depth_string(node[0], d+1) + \" \" + tree_to_depth_string(node[1], d+1)\n    return str(d)",
         critique="the depth was not incremented, so all leaves report depth 0; pass d+1.",
         tests=['tree_to_depth_string((1,2)) == "1 1"',
                'tree_to_depth_string(5) == "0"',
                'tree_to_depth_string((1,(2,3))) == "1 2 2"',
                'tree_to_depth_string(((1,2),3)) == "2 2 1"',
                'tree_to_depth_string((((1,2),3),4)) == "3 3 2 1"']),

    dict(name="parse_paren", pattern="deserialization",
         desc="Write `parse_paren(s)` that parses a lisp-style string (leaf=digits, "
              "branch=\"(L R)\" with a single space) into a nested tuple.",
         wrong="def parse_paren(s):\n    if s[0] == '(':\n        inner = s[1:-1]\n        l, r = inner.split(' ')\n        return (int(l), int(r))\n    return int(s)",
         correct="def parse_paren(s):\n    def parse(i):\n        if s[i] == '(':\n            left, i = parse(i + 1)\n            right, i = parse(i + 1)\n            return (left, right), i + 1\n        j = i\n        while j < len(s) and (s[j].isdigit() or (s[j] == '-' and j == i)):\n            j += 1\n        return int(s[i:j]), j\n    return parse(0)[0]",
         critique="splitting on a single space breaks on nested branches; use a recursive-descent parser tracking the index.",
         tests=['parse_paren("(1 2)") == (1,2)',
                'parse_paren("7") == 7',
                'parse_paren("(1 (2 3))") == (1,(2,3))',
                'parse_paren("((1 2) 3)") == ((1,2),3)',
                'parse_paren("(((1 2) 3) 4)") == (((1,2),3),4)']),

    dict(name="zip_leaves", pattern="reconstruction",
         desc="Two trees `a` and `b` have the SAME shape (nested tuples; leaf=int). Write "
              "`zip_leaves(a, b)` returning a tree of the same shape whose leaves are the sums "
              "of the corresponding leaves of a and b.",
         wrong="def zip_leaves(a, b):\n    if isinstance(a, tuple):\n        return (zip_leaves(a[0], b[0]), zip_leaves(a[1], b[1]))\n    return a",
         correct="def zip_leaves(a, b):\n    if isinstance(a, tuple):\n        return (zip_leaves(a[0], b[0]), zip_leaves(a[1], b[1]))\n    return a + b",
         critique="leaves returned a alone instead of a+b; add the corresponding leaves.",
         tests=['zip_leaves((1,2),(3,4)) == (4,6)',
                'zip_leaves(5,7) == 12',
                'zip_leaves((1,(2,3)),(4,(5,6))) == (5,(7,9))',
                'zip_leaves(((1,2),3),((10,20),30)) == ((11,22),33)',
                'zip_leaves((1,2),(0,0)) == (1,2)']),

    dict(name="tree_map_add", pattern="reconstruction",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_map_add(node, k)` returning a tree of the same shape with k added to every "
              "leaf.",
         wrong="def tree_map_add(node, k):\n    if isinstance(node, tuple):\n        return (tree_map_add(node[0], k), tree_map_add(node[1], k))\n    return node",
         correct="def tree_map_add(node, k):\n    if isinstance(node, tuple):\n        return (tree_map_add(node[0], k), tree_map_add(node[1], k))\n    return node + k",
         critique="leaves were returned unchanged; add k at the base case.",
         tests=['tree_map_add((1,2),10) == (11,12)',
                'tree_map_add(5,3) == 8',
                'tree_map_add((1,(2,3)),1) == (2,(3,4))',
                'tree_map_add(((1,2),3),0) == ((1,2),3)',
                'tree_map_add((1,2),-1) == (0,1)']),

    dict(name="tree_replace_leaf", pattern="reconstruction",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_replace_leaf(node, old, new)` returning a tree of the same shape where every "
              "leaf equal to `old` becomes `new`.",
         wrong="def tree_replace_leaf(node, old, new):\n    if isinstance(node, tuple):\n        return (tree_replace_leaf(node[0], old, new), tree_replace_leaf(node[1], old, new))\n    return new",
         correct="def tree_replace_leaf(node, old, new):\n    if isinstance(node, tuple):\n        return (tree_replace_leaf(node[0], old, new), tree_replace_leaf(node[1], old, new))\n    return new if node == old else node",
         critique="every leaf was replaced unconditionally; only replace leaves equal to old.",
         tests=['tree_replace_leaf((1,2),1,9) == (9,2)',
                'tree_replace_leaf(1,1,5) == 5',
                'tree_replace_leaf((1,(2,1)),1,0) == (0,(2,0))',
                'tree_replace_leaf(((3,2),3),3,7) == ((7,2),7)',
                'tree_replace_leaf((1,2),5,9) == (1,2)']),

    dict(name="tree_height_edges", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_height_edges(node)` returning the height in EDGES (a single leaf has "
              "height 0).",
         wrong="def tree_height_edges(node):\n    if isinstance(node, tuple):\n        return 1 + max(tree_height_edges(node[0]), tree_height_edges(node[1]))\n    return 1",
         correct="def tree_height_edges(node):\n    if isinstance(node, tuple):\n        return 1 + max(tree_height_edges(node[0]), tree_height_edges(node[1]))\n    return 0",
         critique="the leaf base case returned 1 (counting nodes, not edges); a leaf has 0 edges.",
         tests=['tree_height_edges(5) == 0',
                'tree_height_edges((1,2)) == 1',
                'tree_height_edges((1,(2,3))) == 2',
                'tree_height_edges(((1,2),(3,4))) == 2',
                'tree_height_edges((((1,2),3),4)) == 3']),

    dict(name="tree_count_value", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_count_value(node, v)` returning how many leaves equal v.",
         wrong="def tree_count_value(node, v):\n    if isinstance(node, tuple):\n        return tree_count_value(node[0], v) + tree_count_value(node[1], v)\n    return 1",
         correct="def tree_count_value(node, v):\n    if isinstance(node, tuple):\n        return tree_count_value(node[0], v) + tree_count_value(node[1], v)\n    return 1 if node == v else 0",
         critique="every leaf was counted; count only leaves equal to v.",
         tests=['tree_count_value((1,1),1) == 2',
                'tree_count_value(5,5) == 1',
                'tree_count_value((1,(2,1)),1) == 2',
                'tree_count_value((1,2),9) == 0',
                'tree_count_value(((3,3),3),3) == 3']),

    dict(name="tree_internal_count", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_internal_count(node)` returning the number of internal (branch) nodes.",
         wrong="def tree_internal_count(node):\n    if isinstance(node, tuple):\n        return 1 + tree_internal_count(node[0]) + tree_internal_count(node[1])\n    return 1",
         correct="def tree_internal_count(node):\n    if isinstance(node, tuple):\n        return 1 + tree_internal_count(node[0]) + tree_internal_count(node[1])\n    return 0",
         critique="leaves were counted as internal nodes; a leaf contributes 0.",
         tests=['tree_internal_count(5) == 0',
                'tree_internal_count((1,2)) == 1',
                'tree_internal_count((1,(2,3))) == 2',
                'tree_internal_count(((1,2),(3,4))) == 3',
                'tree_internal_count((((1,2),3),4)) == 3']),

    dict(name="tree_product", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_product(node)` returning the product of all leaf values.",
         wrong="def tree_product(node):\n    if isinstance(node, tuple):\n        return tree_product(node[0]) + tree_product(node[1])\n    return node",
         correct="def tree_product(node):\n    if isinstance(node, tuple):\n        return tree_product(node[0]) * tree_product(node[1])\n    return node",
         critique="subtotals were added instead of multiplied; combine with multiplication.",
         tests=['tree_product(5) == 5',
                'tree_product((2,3)) == 6',
                'tree_product((2,(3,4))) == 24',
                'tree_product(((1,2),(3,4))) == 24',
                'tree_product((2,(5,1))) == 10']),

    dict(name="tree_even_leaf_sum", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_even_leaf_sum(node)` returning the sum of the even-valued leaves.",
         wrong="def tree_even_leaf_sum(node):\n    if isinstance(node, tuple):\n        return tree_even_leaf_sum(node[0]) + tree_even_leaf_sum(node[1])\n    return node",
         correct="def tree_even_leaf_sum(node):\n    if isinstance(node, tuple):\n        return tree_even_leaf_sum(node[0]) + tree_even_leaf_sum(node[1])\n    return node if node % 2 == 0 else 0",
         critique="all leaves were summed; include only even leaves.",
         tests=['tree_even_leaf_sum((2,3)) == 2',
                'tree_even_leaf_sum(4) == 4',
                'tree_even_leaf_sum(5) == 0',
                'tree_even_leaf_sum((2,(4,5))) == 6',
                'tree_even_leaf_sum(((1,2),(3,6))) == 8']),

    dict(name="tree_deepest_value", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_deepest_value(node)` returning the value of the deepest leaf (leftmost on a "
              "tie).",
         wrong="def tree_deepest_value(node):\n    if isinstance(node, tuple):\n        return tree_deepest_value(node[0])\n    return node",
         correct="def _dv(n, d):\n    if isinstance(n, tuple):\n        lv, ld = _dv(n[0], d+1)\n        rv, rd = _dv(n[1], d+1)\n        return (lv, ld) if ld >= rd else (rv, rd)\n    return n, d\ndef tree_deepest_value(node):\n    return _dv(node, 0)[0]",
         critique="always recursing left ignores depth; track (value, depth) and keep the deeper one, left on ties.",
         tests=['tree_deepest_value(5) == 5',
                'tree_deepest_value((1,(2,3))) == 2',
                'tree_deepest_value(((1,2),3)) == 1',
                'tree_deepest_value((1,2)) == 1',
                'tree_deepest_value((1,(2,(3,4)))) == 3']),
]


def build_records(tasks):
    records = []
    for t in tasks:
        for test in t["tests"]:
            ok_c, _ = _verify(t["correct"], test)
            ok_w, _ = _verify(t["wrong"], test)
            if not ok_c:
                print(f"ERROR: correct failed for {t['name']} :: {test}", file=sys.stderr)
                sys.exit(1)
            if ok_w:
                continue
            wrong_call = ('TOOL_CALL: execute_code({"code": '
                          + json.dumps(t["wrong"] + "\nassert " + test + "\nprint('PASS')") + "})")
            correct_call = ('TOOL_CALL: execute_code({"code": '
                            + json.dumps(t["correct"] + "\nassert " + test + "\nprint('PASS')") + "})")
            records.append({
                "instruction": f"Original task: {t['desc']}\n\nPrevious failed attempt:\n{wrong_call}",
                "response": (f"CRITIQUE:\n  Correctness — {t['critique']}\n  → Fix needed: correct "
                             f"the {t['pattern']} logic and re-verify.\n{correct_call}\n"
                             f"OBSERVATION: PASS\nCRITIQUE:\n  Correctness — all assertions passed "
                             f"→ Solution OK.\nFINAL_ANSWER: Implemented and verified `{t['name']}`."),
                "task_family": "tree", "pattern": t["pattern"], "func_name": t["name"]})
    return records


def main():
    bench = benchmark_callables()
    all_tasks = BASE_TASKS + EXTRA_TASKS
    names = [t["name"] for t in all_tasks]
    if len(set(names)) != len(names):
        from collections import Counter
        dup = [n for n, c in Counter(names).items() if c > 1]
        print(f"ERROR: duplicate task names: {dup}", file=sys.stderr)
        sys.exit(1)
    overlap = sorted(set(names) & bench)
    if overlap:
        print(f"ERROR: task names overlap benchmark: {overlap}", file=sys.stderr)
        sys.exit(1)

    records = build_records(all_tasks)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    pat = Counter(r["pattern"] for r in records)
    print(f"[v223b] verified {len(records)} records ({len(all_tasks)} distinct tasks: "
          f"{len(BASE_TASKS)} base + {len(EXTRA_TASKS)} new) -> {OUT_PATH}")
    print(f"[v223b] patterns: {dict(pat)}")
    print(f"[v223b] contamination guard PASSED (0 name overlap with {len(bench)} benchmark callables)")


if __name__ == "__main__":
    main()
