"""
scripts/build_v233_scaffold_dataset.py — v2.33 scaffold-first preservation dataset (source-only).

v2.31 (repair-only) and v2.32 (repair + preservation) both improved local repair but collapsed the
frozen 32-task agent benchmark with `no_tool_call` dominant. v2.33 isolates the failure mode: it
builds a dataset of ONLY correct tool-call / execute_code scaffold trajectories — NO repair objective,
NO failed-candidate examples — to test whether tool-use and the 32-task benchmark can be preserved
before any repair adaptation is reintroduced.

Each example is correct agent behaviour: task/instruction → structured tool call (execute_code) →
VERIFIER PASS → verified FINAL_ANSWER, built from NON-held-out reference functions (names disjoint
from the 32-task benchmark; any overlap is rejected).

Output (LOCAL-ONLY, gitignored):
    data/generated/v233/scaffold_train.jsonl
    data/generated/v233/scaffold_val.jsonl
    data/generated/v233/scaffold_aggregate.json   (small; consumed by the summariser)

Usage:
    python scripts/build_v233_scaffold_dataset.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import _contamination_guard, _load_overlap_corpus  # noqa: E402
from scripts.build_v230_broadened_repair_harvest import ALGO_TASKS, FORMAT_TASKS  # noqa: E402
from scripts.build_v226_representation_tasks import TASKS as V226_TASKS  # noqa: E402
from scripts.build_v232_mixed_dataset import _preservation_example  # noqa: E402

OUT_DIR = ROOT / "data" / "generated" / "v233"
VAL_EVERY = 5


def main():
    bench = _load_overlap_corpus()[0]
    corpus = _load_overlap_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = ([(f, fam, ref, tests) for f, fam, ref, tests, _ in FORMAT_TASKS] +
               [(f, fam, ref, tests) for f, fam, ref, tests, _ in ALGO_TASKS] +
               [(f, "tree_serialize_repr", ref, tests) for f, _lt, _rep, _d, ref, tests in V226_TASKS])

    examples, fams, rejected = [], Counter(), Counter()
    for func, family, ref_code, tests in sources:
        if func in bench:
            rejected["heldout_function_name"] += 1
            continue
        cg = _contamination_guard(func, func, ref_code, ref_code, {}, corpus)
        if any(cg.values()):
            rejected["contamination_overlap"] += 1
            continue
        inp, out = _preservation_example(func, family, ref_code, tests)
        examples.append({"objective": "tool_use_preservation", "loss_weight": 1.0,
                         "input": inp, "output": out, "task_family": family, "func": func,
                         "source": "v233_scaffold"})
        fams[family] += 1

    train = [ex for i, ex in enumerate(examples) if i % VAL_EVERY != 0]
    val = [ex for i, ex in enumerate(examples) if i % VAL_EVERY == 0]
    for name, rows in (("scaffold_train", train), ("scaffold_val", val)):
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    objectives = Counter(ex["objective"] for ex in examples)
    agg = {
        "total": len(examples), "train_records": len(train), "val_records": len(val),
        "objective_distribution": dict(objectives),
        "repair_examples": objectives.get("repair", 0),          # must be 0 (scaffold-first)
        "scaffold_examples": objectives.get("tool_use_preservation", 0),
        "task_family_distribution": dict(fams),
        "rejection_reasons": dict(rejected),
        "contamination_guard_violations": 0,   # any overlap rejected above
    }
    (OUT_DIR / "scaffold_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v233] scaffold dataset -> {OUT_DIR} train={len(train)} val={len(val)}  (LOCAL-ONLY)")
    print(f"[v233] objectives={dict(objectives)} (repair MUST be 0) families={len(fams)}")
    print(f"[v233] rejected={dict(rejected)} contamination=0")
    if objectives.get("repair", 0) != 0:
        print("[v233] ERROR: repair examples present — v2.33 is scaffold-first.", file=sys.stderr); sys.exit(2)
    if not examples:
        print("[v233] ERROR: no scaffold examples built.", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
