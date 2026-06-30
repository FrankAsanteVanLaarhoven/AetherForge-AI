"""
scripts/build_v226_representation_tasks.py — v2.26 tree_serialize representation attack.

tree_serialize (exact string) is the lone holdout that never converted across coverage,
planning, verifier repair, targeted adaptation, and scale. This builds a controlled
diagnostic: the SAME three logical tree-serializations (full structure, leaf values,
leaf+depth) expressed in FOUR output representations (exact_string, token_list, nested_list,
json). Holding the algorithm constant and varying only the output FORMAT isolates whether the
difficulty is the exact-string FORMAT (output/control-bound) or the traversal (capability-bound).

All tasks use the same nested-tuple representation (leaf=int, branch=(left,right)), are
execution-verified, and have function names disjoint from the 32-task benchmark. The held-out
benchmark tasks are not touched. Output (eval benchmark): data/v226_representation_tasks.jsonl

Usage:
    python scripts/build_v226_representation_tasks.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402

OUT_PATH = ROOT / "data" / "v226_representation_tasks.jsonl"

# (func_name, logical_task, representation, desc, reference_code, tests)
TASKS = [
    # ── L1: full structural serialization ─────────────────────────────────────
    ("tree_struct_str", "full_structure", "exact_string",
     "Write `tree_struct_str(node)` serializing a nested-tuple tree (leaf=int, "
     "branch=(left,right)) to a lisp-style string: a leaf is str(value), a branch is "
     "\"(\"+left+\" \"+right+\")\".",
     "def tree_struct_str(node):\n    if isinstance(node, tuple):\n        return \"(\" + tree_struct_str(node[0]) + \" \" + tree_struct_str(node[1]) + \")\"\n    return str(node)",
     ['tree_struct_str((1,(2,3))) == "(1 (2 3))"', 'tree_struct_str(5) == "5"',
      'tree_struct_str(((1,2),3)) == "((1 2) 3)"']),

    ("tree_struct_tokens", "full_structure", "token_list",
     "Write `tree_struct_tokens(node)` serializing a nested-tuple tree to a flat list of "
     "string tokens in preorder: a leaf yields [str(value)], a branch yields "
     "[\"(\"] + left_tokens + right_tokens + [\")\"].",
     "def tree_struct_tokens(node):\n    if isinstance(node, tuple):\n        return [\"(\"] + tree_struct_tokens(node[0]) + tree_struct_tokens(node[1]) + [\")\"]\n    return [str(node)]",
     ['tree_struct_tokens((1,(2,3))) == [\"(\",\"1\",\"(\",\"2\",\"3\",\")\",\")\"]',
      'tree_struct_tokens(5) == [\"5\"]',
      'tree_struct_tokens((1,2)) == [\"(\",\"1\",\"2\",\")\"]']),

    ("tree_struct_list", "full_structure", "nested_list",
     "Write `tree_struct_list(node)` serializing a nested-tuple tree to the same structure "
     "with tuples replaced by lists (leaves unchanged).",
     "def tree_struct_list(node):\n    if isinstance(node, tuple):\n        return [tree_struct_list(node[0]), tree_struct_list(node[1])]\n    return node",
     ['tree_struct_list((1,(2,3))) == [1,[2,3]]', 'tree_struct_list(5) == 5',
      'tree_struct_list(((1,2),3)) == [[1,2],3]']),

    ("tree_struct_json", "full_structure", "json",
     "Write `tree_struct_json(node)` serializing a nested-tuple tree to nested dicts: a leaf "
     "becomes {\"leaf\": value}, a branch becomes {\"branch\": [left, right]}.",
     "def tree_struct_json(node):\n    if isinstance(node, tuple):\n        return {\"branch\": [tree_struct_json(node[0]), tree_struct_json(node[1])]}\n    return {\"leaf\": node}",
     ['tree_struct_json((1,2)) == {\"branch\": [{\"leaf\": 1}, {\"leaf\": 2}]}',
      'tree_struct_json(5) == {\"leaf\": 5}',
      'tree_struct_json((1,(2,3))) == {\"branch\": [{\"leaf\": 1}, {\"branch\": [{\"leaf\": 2}, {\"leaf\": 3}]}]}']),

    # ── L2: leaf values in order ──────────────────────────────────────────────
    ("tree_leafvals_csv", "leaf_values", "exact_string",
     "Write `tree_leafvals_csv(node)` returning the preorder leaf values as a "
     "comma-separated string.",
     "def tree_leafvals_csv(node):\n    if isinstance(node, tuple):\n        return tree_leafvals_csv(node[0]) + \",\" + tree_leafvals_csv(node[1])\n    return str(node)",
     ['tree_leafvals_csv((1,(2,3))) == "1,2,3"', 'tree_leafvals_csv(5) == "5"',
      'tree_leafvals_csv(((1,2),3)) == "1,2,3"']),

    ("tree_leafvals_tokens", "leaf_values", "token_list",
     "Write `tree_leafvals_tokens(node)` returning the preorder leaf values as a list of "
     "strings.",
     "def tree_leafvals_tokens(node):\n    if isinstance(node, tuple):\n        return tree_leafvals_tokens(node[0]) + tree_leafvals_tokens(node[1])\n    return [str(node)]",
     ['tree_leafvals_tokens((1,(2,3))) == [\"1\",\"2\",\"3\"]',
      'tree_leafvals_tokens(5) == [\"5\"]',
      'tree_leafvals_tokens((1,2)) == [\"1\",\"2\"]']),

    ("tree_leafvals_list", "leaf_values", "nested_list",
     "Write `tree_leafvals_list(node)` returning the preorder leaf values as a flat list of "
     "ints.",
     "def tree_leafvals_list(node):\n    if isinstance(node, tuple):\n        return tree_leafvals_list(node[0]) + tree_leafvals_list(node[1])\n    return [node]",
     ['tree_leafvals_list((1,(2,3))) == [1,2,3]', 'tree_leafvals_list(5) == [5]',
      'tree_leafvals_list(((1,2),3)) == [1,2,3]']),

    ("tree_leafvals_json", "leaf_values", "json",
     "Write `tree_leafvals_json(node)` returning {\"leaves\": [preorder leaf values]}.",
     "def _lv(node):\n    if isinstance(node, tuple):\n        return _lv(node[0]) + _lv(node[1])\n    return [node]\ndef tree_leafvals_json(node):\n    return {\"leaves\": _lv(node)}",
     ['tree_leafvals_json((1,(2,3))) == {\"leaves\": [1,2,3]}',
      'tree_leafvals_json(5) == {\"leaves\": [5]}',
      'tree_leafvals_json((1,2)) == {\"leaves\": [1,2]}']),

    # ── L3: leaf value with depth ─────────────────────────────────────────────
    ("tree_lvldepth_csv", "leaf_depth", "exact_string",
     "Write `tree_lvldepth_csv(node)` returning preorder leaves as \"value:depth\" pairs "
     "joined by commas (root depth 0).",
     "def tree_lvldepth_csv(node, d=0):\n    if isinstance(node, tuple):\n        return tree_lvldepth_csv(node[0], d+1) + \",\" + tree_lvldepth_csv(node[1], d+1)\n    return str(node) + \":\" + str(d)",
     ['tree_lvldepth_csv((1,(2,3))) == "1:1,2:2,3:2"', 'tree_lvldepth_csv(5) == "5:0"',
      'tree_lvldepth_csv((1,2)) == "1:1,2:1"']),

    ("tree_lvldepth_tokens", "leaf_depth", "token_list",
     "Write `tree_lvldepth_tokens(node)` returning preorder leaves as a list of "
     "\"value:depth\" strings (root depth 0).",
     "def tree_lvldepth_tokens(node, d=0):\n    if isinstance(node, tuple):\n        return tree_lvldepth_tokens(node[0], d+1) + tree_lvldepth_tokens(node[1], d+1)\n    return [str(node) + \":\" + str(d)]",
     ['tree_lvldepth_tokens((1,(2,3))) == [\"1:1\",\"2:2\",\"3:2\"]',
      'tree_lvldepth_tokens(5) == [\"5:0\"]',
      'tree_lvldepth_tokens((1,2)) == [\"1:1\",\"2:1\"]']),

    ("tree_lvldepth_list", "leaf_depth", "nested_list",
     "Write `tree_lvldepth_list(node)` returning preorder leaves as a list of [value, depth] "
     "pairs (root depth 0).",
     "def tree_lvldepth_list(node, d=0):\n    if isinstance(node, tuple):\n        return tree_lvldepth_list(node[0], d+1) + tree_lvldepth_list(node[1], d+1)\n    return [[node, d]]",
     ['tree_lvldepth_list((1,(2,3))) == [[1,1],[2,2],[3,2]]',
      'tree_lvldepth_list(5) == [[5,0]]',
      'tree_lvldepth_list((1,2)) == [[1,1],[2,1]]']),

    ("tree_lvldepth_json", "leaf_depth", "json",
     "Write `tree_lvldepth_json(node)` returning preorder leaves as a list of "
     "{\"v\": value, \"d\": depth} dicts (root depth 0).",
     "def tree_lvldepth_json(node, d=0):\n    if isinstance(node, tuple):\n        return tree_lvldepth_json(node[0], d+1) + tree_lvldepth_json(node[1], d+1)\n    return [{\"v\": node, \"d\": d}]",
     ['tree_lvldepth_json((1,2)) == [{\"v\":1,\"d\":1},{\"v\":2,\"d\":1}]',
      'tree_lvldepth_json(5) == [{\"v\":5,\"d\":0}]',
      'tree_lvldepth_json((1,(2,3))) == [{\"v\":1,\"d\":1},{\"v\":2,\"d\":2},{\"v\":3,\"d\":2}]']),
]


def _verify(code, tests):
    prog = code + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"; fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return (r.returncode == 0 and "PASS" in r.stdout), (r.stdout + r.stderr).strip()


def main():
    bench = benchmark_callables()
    names = [t[0] for t in TASKS]
    overlap = sorted(set(names) & bench) + sorted(n for n in bench if any(n in x for x in names))
    if overlap:
        print(f"ERROR: representation task names overlap/contain benchmark callables: {overlap}", file=sys.stderr)
        sys.exit(1)

    records = []
    for func, logical, rep, desc, ref, tests in TASKS:
        ok, out = _verify(ref, tests)
        print(f"  [{'PASS' if ok else 'FAIL'}] {rep:12} {func}")
        if not ok:
            print(f"      {out[:300]}", file=sys.stderr); sys.exit(1)
        verify_text = "; ".join(tests)
        prompt = (f"{desc} Verify with assertions and end with print('PASS'): "
                  f"{verify_text}.")
        records.append({"id": f"v226_{func}", "family": "tree_serialize_repr",
                        "category": rep, "prompt": prompt,
                        "representation": rep, "logical_task": logical, "func_name": func})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    rc = Counter(r["representation"] for r in records)
    print(f"\n[v226] verified {len(records)} representation tasks -> {OUT_PATH}")
    print(f"[v226] representations: {dict(rc)}")
    print(f"[v226] 3 logical serializations x 4 formats. contamination guard PASSED "
          f"(0 overlap with {len(bench)} benchmark callables)")


if __name__ == "__main__":
    main()
