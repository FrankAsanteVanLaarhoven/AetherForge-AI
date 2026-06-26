"""
scripts/check_v223_contamination.py — v2.23 benchmark-contamination guard.

Asserts the v2.23 training data (data/v223_tree_capability_records.jsonl) does NOT leak the
held-out 32-task benchmark. Aborts (exit 1) on any overlap. The three hard tree tasks
(tree_serialize, tree_from_list, tree_max_path_sum) must remain evaluation-only.

Checks:
  1. 0 training function-name overlap with the benchmark callables.
  2. No benchmark callable name appears anywhere in any training record (instruction/response).
  3. No benchmark prompt appears (as a substring) in any training instruction.
  4. The three hard task ids / function names do not appear in the dataset.

Usage:
    python scripts/check_v223_contamination.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v219b_family_records import benchmark_callables  # noqa: E402

TRAIN = ROOT / "data" / "v223_tree_capability_records.jsonl"
BENCH = ROOT / "data" / "v210_clean_repair_generalisation_tasks.jsonl"
HARD = ["tree_serialize", "tree_from_list", "tree_max_path_sum",
        "tree_to_str", "tree_from_sorted", "tree_max_path"]


def main():
    if not TRAIN.exists():
        print(f"ERROR: {TRAIN} missing — run build_v223_tree_capability_data.py first.", file=sys.stderr)
        sys.exit(1)
    records = [json.loads(l) for l in open(TRAIN) if l.strip()]
    bench_names = benchmark_callables()
    bench_prompts = []
    for line in open(BENCH):
        line = line.strip()
        if line:
            r = json.loads(line)
            bench_prompts.append((r.get("prompt") or r.get("task", "")).strip())

    failures = []

    # 1 + 4: function-name disjointness (incl. hard tasks)
    train_funcs = {r.get("func_name", "") for r in records}
    name_overlap = sorted(train_funcs & bench_names)
    if name_overlap:
        failures.append(f"function-name overlap with benchmark: {name_overlap}")

    # 2: no benchmark callable name appears verbatim in any training text
    blob = "\n".join((r.get("instruction", "") + "\n" + r.get("response", "")) for r in records)
    leaked = sorted(n for n in bench_names if n in blob)
    if leaked:
        failures.append(f"benchmark callable name(s) appear in training text: {leaked}")

    # 3: no benchmark prompt is a substring of any training instruction
    for r in records:
        instr = r.get("instruction", "")
        for bp in bench_prompts:
            if bp and bp in instr:
                failures.append(f"benchmark prompt leaked into training instruction: {bp[:60]!r}")
                break

    # 4 (explicit): hard task tokens absent
    hard_hits = sorted(h for h in HARD if h in blob)
    if hard_hits:
        failures.append(f"hard held-out task token(s) present in training data: {hard_hits}")

    print(f"[v223-contam] records={len(records)} train_funcs={len(train_funcs)} "
          f"benchmark_callables={len(bench_names)}")
    if failures:
        print("CONTAMINATION CHECK FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("[v223-contam] PASSED: 0 name overlap, 0 benchmark-name leakage, 0 prompt leakage, "
          "0 hard-task tokens. Held-out tasks remain evaluation-only.")


if __name__ == "__main__":
    main()
