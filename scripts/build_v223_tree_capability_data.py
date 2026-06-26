"""
scripts/build_v223_tree_capability_data.py — v2.23 targeted tree-capability training data.

v2.22/v2.22b confirmed three tree tasks are capability-bound (tree_serialize, tree_from_list,
tree_max_path_sum) — they fail under coverage, planning, AND verifier-guided repair. This
builds a small, CONTAMINATION-GUARDED training set that exercises the SAME capability patterns
on DIFFERENT tasks:

  - recursive serialization / deserialization  (vs tree_serialize)
  - recursive structure reconstruction         (vs tree_from_list)
  - two-value recursive tree DP                (vs tree_max_path_sum)

All tasks use the SAME nested-tuple representation (leaf=int, branch=(left,right)) so the
recursive-coding capability can transfer, but every function name / task is DIFFERENT from the
benchmark, every solution is execution-verified, and the wrong attempt genuinely fails. Records
are emitted as {instruction, response} repair trajectories matching the finetune format.

The held-out hard tasks stay evaluation-only. Output (committed, reviewable source):
data/v223_tree_capability_records.jsonl

Usage:
    python scripts/build_v223_tree_capability_data.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402

OUT_PATH = ROOT / "data" / "v223_tree_capability_records.jsonl"

# Each task: name, pattern, plain_instruction, wrong_code (plausible bug), correct_code,
# critique (bug diagnosis), and a list of test-assert variants (each a verified check).
TASKS = [
    # ── recursive serialization / repr (vs tree_serialize) ───────────────────
    dict(name="tree_to_bracket", pattern="serialization",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_to_bracket(node)` returning a bracketed string: a leaf is str(value), a "
              "branch is \"[\"+left+\",\"+right+\"]\".",
         wrong="def tree_to_bracket(node):\n    if isinstance(node, tuple):\n        return tree_to_bracket(node[0]) + tree_to_bracket(node[1])\n    return str(node)",
         correct="def tree_to_bracket(node):\n    if isinstance(node, tuple):\n        return \"[\" + tree_to_bracket(node[0]) + \",\" + tree_to_bracket(node[1]) + \"]\"\n    return str(node)",
         critique="the recursive case dropped the brackets and comma, so structure is lost; wrap left/right in \"[\",\",\",\"]\".",
         tests=['tree_to_bracket((1,(2,3))) == "[1,[2,3]]"',
                'tree_to_bracket(7) == "7"',
                'tree_to_bracket(((1,2),(3,4))) == "[[1,2],[3,4]]"',
                'tree_to_bracket((1,2)) == "[1,2]"',
                'tree_to_bracket((((1,2),3),4)) == "[[[1,2],3],4]"']),

    dict(name="tree_depth_pairs", pattern="serialization",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_depth_pairs(node)` returning a list of (depth, value) for each leaf, "
              "left to right, root depth 0.",
         wrong="def tree_depth_pairs(node, depth=0):\n    if isinstance(node, tuple):\n        return tree_depth_pairs(node[0], depth) + tree_depth_pairs(node[1], depth)\n    return [(depth, node)]",
         correct="def tree_depth_pairs(node, depth=0):\n    if isinstance(node, tuple):\n        return tree_depth_pairs(node[0], depth+1) + tree_depth_pairs(node[1], depth+1)\n    return [(depth, node)]",
         critique="the depth counter was not incremented when recursing into branches, so every leaf reports depth 0; pass depth+1.",
         tests=['tree_depth_pairs((1,(2,3))) == [(1,1),(2,2),(2,3)]',
                'tree_depth_pairs(9) == [(0,9)]',
                'tree_depth_pairs((1,2)) == [(1,1),(1,2)]',
                'tree_depth_pairs(((1,2),3)) == [(2,1),(2,2),(1,3)]',
                'tree_depth_pairs((((4,5),6),7)) == [(3,4),(3,5),(2,6),(1,7)]']),

    dict(name="parse_bracket", pattern="deserialization",
         desc="Write `parse_bracket(s)` that parses a bracketed string (leaf=digits, "
              "branch=\"[L,R]\") back into a nested tuple (leaf=int, branch=(left,right)).",
         wrong="def parse_bracket(s):\n    if s[0] == '[':\n        inner = s[1:-1]\n        l, r = inner.split(',')\n        return (int(l), int(r))\n    return int(s)",
         correct="def parse_bracket(s):\n    def parse(i):\n        if s[i] == '[':\n            left, i = parse(i + 1)\n            right, i = parse(i + 1)\n            return (left, right), i + 1\n        j = i\n        while j < len(s) and (s[j].isdigit() or (s[j] == '-' and j == i)):\n            j += 1\n        return int(s[i:j]), j\n    return parse(0)[0]",
         critique="splitting on ',' breaks on nested brackets; use a recursive-descent parser that tracks the index.",
         tests=['parse_bracket("[1,[2,3]]") == (1,(2,3))',
                'parse_bracket("7") == 7',
                'parse_bracket("[[1,2],[3,4]]") == ((1,2),(3,4))',
                'parse_bracket("[1,2]") == (1,2)',
                'parse_bracket("[[[1,2],3],4]") == (((1,2),3),4)']),

    # ── structure reconstruction (vs tree_from_list) ─────────────────────────
    dict(name="build_left_spine", pattern="reconstruction",
         desc="Write `build_left_spine(nums)` building a left-leaning nested-tuple tree: "
              "fold so each new value attaches on the right of the growing left subtree.",
         wrong="def build_left_spine(nums):\n    node = nums[0]\n    for v in nums[1:]:\n        node = (v, node)\n    return node",
         correct="def build_left_spine(nums):\n    node = nums[0]\n    for v in nums[1:]:\n        node = (node, v)\n    return node",
         critique="the accumulator was placed on the right child, giving a right spine; put the growing subtree on the left: (node, v).",
         tests=['build_left_spine([1,2,3]) == ((1,2),3)',
                'build_left_spine([5]) == 5',
                'build_left_spine([1,2]) == (1,2)',
                'build_left_spine([1,2,3,4]) == (((1,2),3),4)',
                'build_left_spine([9,8,7]) == ((9,8),7)']),

    dict(name="build_right_spine", pattern="reconstruction",
         desc="Write `build_right_spine(nums)` building a right-leaning nested-tuple tree: "
              "each value attaches as the left child of the growing right subtree.",
         wrong="def build_right_spine(nums):\n    node = nums[-1]\n    for v in nums[:-1]:\n        node = (v, node)\n    return node",
         correct="def build_right_spine(nums):\n    node = nums[-1]\n    for v in reversed(nums[:-1]):\n        node = (v, node)\n    return node",
         critique="iterating the prefix left-to-right reverses the nesting order; iterate it reversed.",
         tests=['build_right_spine([1,2,3]) == (1,(2,3))',
                'build_right_spine([5]) == 5',
                'build_right_spine([1,2]) == (1,2)',
                'build_right_spine([1,2,3,4]) == (1,(2,(3,4)))',
                'build_right_spine([7,8,9]) == (7,(8,9))']),

    # ── two-value recursive tree DP (vs tree_max_path_sum) ────────────────────
    dict(name="tree_diameter", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_diameter(node)` returning the longest path length (number of edges) "
              "between any two leaves.",
         wrong="def tree_diameter(node):\n    if isinstance(node, tuple):\n        return tree_diameter(node[0]) + tree_diameter(node[1])\n    return 0",
         correct="def _h(node):\n    if isinstance(node, tuple):\n        lh, ld = _h(node[0])\n        rh, rd = _h(node[1])\n        return 1 + max(lh, rh), max(ld, rd, lh + rh + 2)\n    return 0, 0\ndef tree_diameter(node):\n    return _h(node)[1]",
         critique="diameter is not the sum of child diameters; track height and diameter together and combine the two leaf-to-node edge counts (lh+rh+2) through the node.",
         tests=['tree_diameter(5) == 0',
                'tree_diameter((1,2)) == 2',
                'tree_diameter(((1,2),(3,4))) == 4',
                'tree_diameter((1,(2,3))) == 3',
                'tree_diameter(((1,(2,3)),4)) == 4']),

    dict(name="tree_best_subtree", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_best_subtree(node)` returning the maximum total of leaf values over any "
              "subtree (a subtree is any node and all its descendants).",
         wrong="def tree_best_subtree(node):\n    if isinstance(node, tuple):\n        return tree_best_subtree(node[0]) + tree_best_subtree(node[1])\n    return node",
         correct="def _ts(node):\n    if isinstance(node, tuple):\n        lt, lm = _ts(node[0])\n        rt, rm = _ts(node[1])\n        total = lt + rt\n        return total, max(total, lm, rm)\n    return node, node\ndef tree_best_subtree(node):\n    return _ts(node)[1]",
         critique="returning the whole-tree total ignores that a smaller subtree may be larger when values are negative; track (subtree_total, best_seen).",
         tests=['tree_best_subtree(5) == 5',
                'tree_best_subtree((1,2)) == 3',
                'tree_best_subtree((3,(-1,-2))) == 3',
                'tree_best_subtree(((1,2),(3,4))) == 10',
                'tree_best_subtree((-5,(2,3))) == 5']),

    dict(name="tree_is_balanced", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_is_balanced(node)` returning True iff every branch's left and right "
              "subtree heights differ by at most 1.",
         wrong="def tree_is_balanced(node):\n    if isinstance(node, tuple):\n        return tree_is_balanced(node[0]) and tree_is_balanced(node[1])\n    return True",
         correct="def _hb(node):\n    if isinstance(node, tuple):\n        lh, lb = _hb(node[0])\n        rh, rb = _hb(node[1])\n        return 1 + max(lh, rh), lb and rb and abs(lh - rh) <= 1\n    return 0, True\ndef tree_is_balanced(node):\n    return _hb(node)[1]",
         critique="checking children alone misses the height imbalance at the current node; return (height, balanced) and test abs(lh-rh)<=1.",
         tests=['tree_is_balanced(5) == True',
                'tree_is_balanced((1,2)) == True',
                'tree_is_balanced((1,(2,3))) == True',
                'tree_is_balanced((((1,2),3),4)) == False',
                'tree_is_balanced(((1,2),((3,4),5))) == True']),

    dict(name="tree_leaf_count", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_leaf_count(node)` returning the number of leaves.",
         wrong="def tree_leaf_count(node):\n    if isinstance(node, tuple):\n        return tree_leaf_count(node[0]) + tree_leaf_count(node[1])\n    return 0",
         correct="def tree_leaf_count(node):\n    if isinstance(node, tuple):\n        return tree_leaf_count(node[0]) + tree_leaf_count(node[1])\n    return 1",
         critique="the base case returned 0 instead of 1, so no leaf is ever counted; a leaf contributes 1.",
         tests=['tree_leaf_count(5) == 1',
                'tree_leaf_count((1,2)) == 2',
                'tree_leaf_count((1,(2,3))) == 3',
                'tree_leaf_count(((1,2),(3,4))) == 4',
                'tree_leaf_count((((1,2),3),(4,5))) == 5']),

    dict(name="tree_value_total", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_value_total(node)` returning the total of all leaf values.",
         wrong="def tree_value_total(node):\n    if isinstance(node, tuple):\n        return tree_value_total(node[0])\n    return node",
         correct="def tree_value_total(node):\n    if isinstance(node, tuple):\n        return tree_value_total(node[0]) + tree_value_total(node[1])\n    return node",
         critique="only the left child was summed; add both left and right subtotals.",
         tests=['tree_value_total(5) == 5',
                'tree_value_total((1,2)) == 3',
                'tree_value_total((1,(2,3))) == 6',
                'tree_value_total(((1,2),(3,4))) == 10',
                'tree_value_total((10,(20,30))) == 60']),

    dict(name="tree_to_paren", pattern="serialization",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_to_paren(node)` returning a lisp-style string: a leaf is str(value), a "
              "branch is \"(\"+left+\" \"+right+\")\".",
         wrong="def tree_to_paren(node):\n    if isinstance(node, tuple):\n        return tree_to_paren(node[0]) + \" \" + tree_to_paren(node[1])\n    return str(node)",
         correct="def tree_to_paren(node):\n    if isinstance(node, tuple):\n        return \"(\" + tree_to_paren(node[0]) + \" \" + tree_to_paren(node[1]) + \")\"\n    return str(node)",
         critique="the parentheses around each branch were dropped, so nesting is ambiguous; wrap each branch in \"(\"...\")\".",
         tests=['tree_to_paren((1,(2,3))) == "(1 (2 3))"',
                'tree_to_paren(7) == "7"',
                'tree_to_paren(((1,2),3)) == "((1 2) 3)"',
                'tree_to_paren((1,2)) == "(1 2)"',
                'tree_to_paren((((1,2),3),4)) == "(((1 2) 3) 4)"']),

    dict(name="tree_max_leaf", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_max_leaf(node)` returning the maximum leaf value.",
         wrong="def tree_max_leaf(node):\n    if isinstance(node, tuple):\n        return tree_max_leaf(node[0])\n    return node",
         correct="def tree_max_leaf(node):\n    if isinstance(node, tuple):\n        return max(tree_max_leaf(node[0]), tree_max_leaf(node[1]))\n    return node",
         critique="only the left subtree was considered; take the max over both subtrees.",
         tests=['tree_max_leaf(5) == 5',
                'tree_max_leaf((1,9)) == 9',
                'tree_max_leaf((1,(2,8))) == 8',
                'tree_max_leaf(((7,2),(3,4))) == 7',
                'tree_max_leaf((10,(20,5))) == 20']),

    dict(name="tree_size", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_size(node)` returning the total number of nodes (leaves and branches).",
         wrong="def tree_size(node):\n    if isinstance(node, tuple):\n        return tree_size(node[0]) + tree_size(node[1])\n    return 1",
         correct="def tree_size(node):\n    if isinstance(node, tuple):\n        return 1 + tree_size(node[0]) + tree_size(node[1])\n    return 1",
         critique="the branch node itself was not counted; add 1 for the branch.",
         tests=['tree_size(5) == 1',
                'tree_size((1,2)) == 3',
                'tree_size((1,(2,3))) == 5',
                'tree_size(((1,2),3)) == 5',
                'tree_size((((1,2),3),4)) == 7']),

    dict(name="tree_left_leaf", pattern="reconstruction",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_left_leaf(node)` returning the value of the leftmost leaf.",
         wrong="def tree_left_leaf(node):\n    if isinstance(node, tuple):\n        return node[0]\n    return node",
         correct="def tree_left_leaf(node):\n    if isinstance(node, tuple):\n        return tree_left_leaf(node[0])\n    return node",
         critique="returning node[0] yields a subtree, not a leaf; recurse into the left child until a leaf is reached.",
         tests=['tree_left_leaf(5) == 5',
                'tree_left_leaf((1,2)) == 1',
                'tree_left_leaf(((7,8),9)) == 7',
                'tree_left_leaf((((1,2),3),4)) == 1',
                'tree_left_leaf((4,(5,6))) == 4']),

    dict(name="tree_min_leaf_value", pattern="path_dp",
         desc="A tree is a nested tuple (leaf=int, branch=(left,right)). Write "
              "`tree_min_leaf_value(node)` returning the minimum leaf value.",
         wrong="def tree_min_leaf_value(node):\n    if isinstance(node, tuple):\n        return tree_min_leaf_value(node[1])\n    return node",
         correct="def tree_min_leaf_value(node):\n    if isinstance(node, tuple):\n        return min(tree_min_leaf_value(node[0]), tree_min_leaf_value(node[1]))\n    return node",
         critique="only the right subtree was considered; take the min over both subtrees.",
         tests=['tree_min_leaf_value(5) == 5',
                'tree_min_leaf_value((9,1)) == 1',
                'tree_min_leaf_value((4,(2,8))) == 2',
                'tree_min_leaf_value(((7,2),(3,4))) == 2',
                'tree_min_leaf_value((10,(20,5))) == 5']),
]


def _verify(code: str, test: str) -> tuple[bool, str]:
    prog = code + "\nassert " + test + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"
        fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return (r.returncode == 0 and "PASS" in r.stdout), (r.stdout + r.stderr).strip()


def main():
    bench = benchmark_callables()
    names = [t["name"] for t in TASKS]
    overlap = sorted(set(names) & bench)
    if overlap:
        print(f"ERROR: training function names overlap benchmark: {overlap}", file=sys.stderr)
        sys.exit(1)

    records = []
    for t in TASKS:
        for test in t["tests"]:
            # correct must pass; wrong must fail (realistic repair).
            ok_c, _ = _verify(t["correct"], test)
            ok_w, _ = _verify(t["wrong"], test)
            if not ok_c:
                print(f"ERROR: correct solution failed for {t['name']} :: {test}", file=sys.stderr)
                sys.exit(1)
            if ok_w:
                # wrong happens to pass this case — skip (not a useful repair example)
                continue
            wrong_call = ('TOOL_CALL: execute_code({"code": '
                          + json.dumps(t["wrong"] + "\nassert " + test + "\nprint('PASS')") + "})")
            correct_call = ('TOOL_CALL: execute_code({"code": '
                            + json.dumps(t["correct"] + "\nassert " + test + "\nprint('PASS')") + "})")
            instruction = (f"Original task: {t['desc']}\n\n"
                           f"Previous failed attempt:\n{wrong_call}")
            response = (
                f"CRITIQUE:\n  Correctness — {t['critique']}\n  → Fix needed: correct the "
                f"{t['pattern']} logic and re-verify.\n"
                f"{correct_call}\n"
                f"OBSERVATION: PASS\n"
                f"CRITIQUE:\n  Correctness — all assertions passed → Solution OK.\n"
                f"FINAL_ANSWER: Implemented and verified `{t['name']}`."
            )
            records.append({"instruction": instruction, "response": response,
                            "task_family": "tree", "pattern": t["pattern"],
                            "func_name": t["name"]})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    pat = Counter(r["pattern"] for r in records)
    print(f"[v223] verified {len(records)} repair-trajectory records -> {OUT_PATH}")
    print(f"[v223] patterns: {dict(pat)}")
    print(f"[v223] distinct tasks: {len(names)} | contamination guard PASSED "
          f"(0 name overlap with {len(bench)} benchmark callables)")


if __name__ == "__main__":
    main()
