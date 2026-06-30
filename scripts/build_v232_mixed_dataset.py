"""
scripts/build_v232_mixed_dataset.py — v2.32 mixed repair + tool-use-preservation dataset (source-only).

v2.31 trains repair behaviour but risks eroding the tool-call / scaffold behaviour the frozen 32-task
benchmark depends on. v2.32 mixes the genuine repair traces with TOOL-USE PRESERVATION traces —
correct agentic tool-call trajectories (PLAN → execute_code → VERIFIER PASS → FINAL_ANSWER) on
NON-held-out tasks — and tags each example with an `objective` + `loss_weight` so the trainer can
apply a SPLIT LOSS (repair objective + tool-use preservation objective). No held-out benchmark task,
solution, or test is used.

Output (LOCAL-ONLY, gitignored):
    data/generated/v232/sft_train.jsonl
    data/generated/v232/sft_val.jsonl
    data/generated/v232/mixed_aggregate.json   (small; consumed by the summariser)

Usage:
    python scripts/build_v232_mixed_dataset.py [--preservation-weight 1.0]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import _contamination_guard, _load_overlap_corpus  # noqa: E402
from scripts.build_v230_broadened_repair_harvest import ALGO_TASKS, FORMAT_TASKS  # noqa: E402
from scripts.build_v226_representation_tasks import TASKS as V226_TASKS  # noqa: E402

V231 = ROOT / "data" / "generated" / "v231"
OUT_DIR = ROOT / "data" / "generated" / "v232"
VAL_EVERY = 5


def _preservation_example(func, family, ref_code, tests):
    """A correct tool-call scaffold trajectory teaching/preserving tool-use format."""
    program = ref_code + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('PASS')"
    inp = (f"INPUT:\n### Task\nImplement `{func}` so the tests pass; use the execute_code tool and "
           f"verify with assertions.\n")
    out = (
        "OUTPUT:\n"
        "PLAN:\n"
        f"  pattern: {func}\n"
        f"  approach: implement the {family} logic, then verify with asserts\n"
        "TOOL_CALL: execute_code({\n"
        f'  "code": "{json.dumps(program)[1:-1]}"\n'
        "})\n"
        "OBSERVATION:\n"
        "VERIFIER:\n  status: PASS\n"
        f"FINAL_ANSWER:\n{ref_code}\n"
    )
    return inp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preservation-weight", type=float, default=1.0,
                    help="per-token loss weight for tool-use preservation examples (split loss)")
    args = ap.parse_args()

    if not (V231 / "sft_train.jsonl").exists():
        print("Run make build-v231-sft-dataset first.")
        sys.exit(1)
    bench = _load_overlap_corpus()[0]
    corpus = _load_overlap_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── repair objective (reuse the v2.31 export) ──
    def _load(split):
        rows = []
        p = V231 / f"sft_{split}.jsonl"
        for line in open(p):
            r = json.loads(line)
            rows.append({"objective": "repair", "loss_weight": 1.0, "input": r["input"],
                         "output": r["output"], "task_family": r.get("task_family", ""),
                         "category": r.get("category", ""), "source": "v231"})
        return rows
    repair_train, repair_val = _load("train"), _load("val")

    # ── tool-use preservation objective (correct scaffolds on non-held-out tasks) ──
    pres = []
    fams = Counter()
    rejected = Counter()
    pres_sources = ([(f, fam, ref, tests) for f, fam, ref, tests, _ in FORMAT_TASKS] +
                    [(f, fam, ref, tests) for f, fam, ref, tests, _ in ALGO_TASKS] +
                    [(f, "tree_serialize_repr", ref, tests) for f, _lt, _rep, _d, ref, tests in V226_TASKS])
    for func, family, ref_code, tests in pres_sources:
        if func in bench:
            rejected["heldout_function_name"] += 1
            continue
        cg = _contamination_guard(func, func, ref_code, ref_code, {}, corpus)
        if any(cg.values()):
            rejected["contamination_overlap"] += 1
            continue
        inp, out = _preservation_example(func, family, ref_code, tests)
        pres.append({"objective": "tool_use_preservation", "loss_weight": args.preservation_weight,
                     "input": inp, "output": out, "task_family": family, "category": "scaffold",
                     "source": "v232_preservation"})
        fams[family] += 1
    pres_train = [ex for i, ex in enumerate(pres) if i % VAL_EVERY != 0]
    pres_val = [ex for i, ex in enumerate(pres) if i % VAL_EVERY == 0]

    train = repair_train + pres_train
    val = repair_val + pres_val
    for name, rows in (("sft_train", train), ("sft_val", val)):
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_repair = sum(1 for ex in train + val if ex["objective"] == "repair")
    n_pres = sum(1 for ex in train + val if ex["objective"] == "tool_use_preservation")
    total = n_repair + n_pres
    agg = {
        "total": total, "train_records": len(train), "val_records": len(val),
        "repair_examples": n_repair, "tool_use_preservation_examples": n_pres,
        "mix_ratio_repair": round(n_repair / total, 3) if total else 0.0,
        "mix_ratio_preservation": round(n_pres / total, 3) if total else 0.0,
        "preservation_loss_weight": args.preservation_weight,
        "objective_distribution": {"repair": n_repair, "tool_use_preservation": n_pres},
        "preservation_family_distribution": dict(fams),
        "rejection_reasons": dict(rejected),
        "contamination_guard_violations": 0,   # any overlap is rejected above
        "val_repair": sum(1 for ex in val if ex["objective"] == "repair"),
        "val_preservation": sum(1 for ex in val if ex["objective"] == "tool_use_preservation"),
    }
    (OUT_DIR / "mixed_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v232] mixed SFT -> {OUT_DIR} train={len(train)} val={len(val)}  (LOCAL-ONLY)")
    print(f"[v232] objectives: repair={n_repair} preservation={n_pres} "
          f"(mix {agg['mix_ratio_repair']}/{agg['mix_ratio_preservation']}, "
          f"pres_loss_weight={args.preservation_weight})")
    print(f"[v232] preservation families={dict(fams)} rejected={dict(rejected)} contamination=0")
    if n_pres == 0:
        print("[v232] ERROR: no preservation traces built.", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
