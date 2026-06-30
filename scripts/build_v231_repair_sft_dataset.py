"""
scripts/build_v231_repair_sft_dataset.py — v2.31 repair-trace SFT dataset export (source-only).

Exports a LOCAL-ONLY supervised fine-tuning dataset from the genuine repair traces (v2.29 + v2.30):
each example maps a failed candidate + structured verifier signal to a repair plan + corrected
solution. Contamination-clean records only; no held-out benchmark task/solution is used.

    INPUT : task instruction + failed candidate + structured verifier signal + failure_type + hint
    OUTPUT: repair plan + corrected solution

Output (LOCAL-ONLY, gitignored):
    data/generated/v231/sft_train.jsonl
    data/generated/v231/sft_val.jsonl
    data/generated/v231/sft_aggregate.json   (small; consumed by the summariser)

Usage:
    python scripts/build_v231_repair_sft_dataset.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import _load_overlap_corpus  # noqa: E402

SOURCES = [("v229", ROOT / "data/generated/v229/repair_traces.jsonl", "format_repair"),
           ("v230", ROOT / "data/generated/v230/repair_traces.jsonl", None)]   # None => use record field
OUT_DIR = ROOT / "data" / "generated" / "v231"
VAL_EVERY = 5   # deterministic ~20% validation hold-out


def _format_example(r):
    v = r.get("verifier_signal", {}) or {}
    func = r.get("logical_task") or r.get("task_id", "")
    rep = r.get("representation", "")
    instruction = f"Repair `{func}` so its output passes the tests (representation: {rep})."
    inp = (
        "INPUT:\n"
        f"### Task\n{instruction}\n"
        f"### Failed candidate\n{r.get('candidate_solution','').strip()}\n"
        "### Verifier signal\n"
        f"status: fail\nfailure_type: {v.get('failure_type','')}\n"
        f"expected: {v.get('expected','')}\nobserved: {v.get('observed','')}\n"
        f"diagnosis: {v.get('diagnosis','')}\nrepair_hint: {v.get('repair_hint','')}\n"
    )
    out = (
        "OUTPUT:\n"
        f"### Repair plan\n{r.get('repair_plan','').strip()}\n"
        f"### Corrected solution\n{r.get('final_solution','').strip()}\n"
    )
    return inp, out


def main():
    bench = _load_overlap_corpus()[0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples, rejected = [], Counter()
    cats, fams = Counter(), Counter()

    for source, path, default_cat in SOURCES:
        if not path.exists():
            continue
        for line in open(path):
            r = json.loads(line)
            # contamination + genuineness guards
            if any((r.get("contamination_guard", {}) or {}).values()):
                rejected["contamination_overlap"] += 1
                continue
            func = r.get("logical_task", "")
            tid = r.get("task_id", "")
            if func in bench or tid in bench:
                rejected["heldout_name_overlap"] += 1
                continue
            if r.get("candidate_solution", "").strip() == r.get("final_solution", "").strip():
                rejected["degenerate_candidate_equals_final"] += 1
                continue
            if r.get("final_status") != "pass":
                rejected["final_not_pass"] += 1
                continue
            if not (r.get("repair_plan") and (r.get("verifier_signal") or {}).get("failure_type")):
                rejected["missing_plan_or_signal"] += 1
                continue
            category = r.get("repair_category") or default_cat or "format_repair"
            inp, out = _format_example(r)
            examples.append({"source": source, "task_id": tid, "category": category,
                             "task_family": r.get("task_family", ""), "input": inp, "output": out})
            cats[category] += 1
            fams[r.get("task_family", "")] += 1

    # deterministic train/val split (every VAL_EVERY-th record to validation), stratified by order
    train, val = [], []
    for i, ex in enumerate(examples):
        (val if i % VAL_EVERY == 0 else train).append(ex)

    for name, rows in (("sft_train", train), ("sft_val", val)):
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    agg = {
        "total_available": len(examples),
        "train_records": len(train),
        "val_records": len(val),
        "format_repair": cats.get("format_repair", 0),
        "algorithmic_repair": cats.get("algorithmic_repair", 0),
        "mixed_repair": cats.get("mixed_repair", 0),
        "category_distribution": dict(cats),
        "task_family_distribution": dict(fams),
        "rejection_reasons": dict(rejected),
        "contamination_guard_violations": 0,   # any overlap is rejected above
        "source_counts": dict(Counter(ex["source"] for ex in examples)),
    }
    (OUT_DIR / "sft_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[v231] SFT export -> {OUT_DIR}/sft_train.jsonl ({len(train)}), sft_val.jsonl ({len(val)})  (LOCAL-ONLY)")
    print(f"[v231] available={len(examples)} format={agg['format_repair']} algo={agg['algorithmic_repair']} "
          f"families={len(fams)}")
    print(f"[v231] rejections={dict(rejected)} contamination_violations=0")
    if not examples:
        print("[v231] ERROR: no repair traces found. Run make build-v229-harvest + build-v230-harvest first.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
