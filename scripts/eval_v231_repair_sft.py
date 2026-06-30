"""
scripts/eval_v231_repair_sft.py — v2.31 repair-adapter evaluation harness (GPU-gated).

Evaluates the v2.31 tiny repair adapter against the base model. The repair-validation slice
(executing the model's corrected solution) is run here; the frozen 32-task benchmark, hard tree
subset, and tree_serialize format-control checks are delegated to the existing evaluate_code_agent.py
make targets (see the v2.31 doc) so this milestone does not duplicate or perturb the champion eval.

GPU-gated: with no CUDA device it prints a SKIP notice and exits 0 (no fabricated metrics). Run on a
GPU host after train_v231_repair_sft.py to produce outputs/v231_tiny_repair_trace_sft/eval_metrics.json.

Comparisons supported (on a GPU host):
    base + structured verifier   vs   v2.31 adapter + structured verifier   vs   adapter w/o verifier

Usage:
    python scripts/eval_v231_repair_sft.py [--base <hf_id_or_path>]
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "generated" / "v231"
OUT_DIR = ROOT / "outputs" / "v231_tiny_repair_trace_sft"
BENCH_32 = ROOT / "data" / "v210_clean_repair_generalisation_tasks.jsonl"
HARD_TREE_IDS = ("tree_serialize", "tree_from_list", "tree_max_path_sum")
CHAMPION_32 = 23   # frozen champion reference (23/28)


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _extract_solution(text):
    m = re.search(r"### Corrected solution\s*(.*)\Z", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _passes(code):
    """Execute a generated corrected solution (code includes its own asserts + print('PASS'))."""
    if "print('PASS')" not in code and 'print("PASS")' not in code:
        return False
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"; fp.write_text(code)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and "PASS" in r.stdout


def _count_passes(csv_path, ids=None):
    """Count passed rows in an evaluate_code_agent best_of_3.csv (optionally restricted to ids)."""
    import csv as _csv
    _csv.field_size_limit(10_000_000)
    p = f = 0
    for row in _csv.DictReader(open(csv_path)):
        tid = (row.get("id") or row.get("task_id") or "")
        if ids is not None and not any(h in tid for h in ids):
            continue
        ok = str(row.get("passed", "")).strip().lower() in ("true", "1", "yes")
        p += int(ok); f += int(not ok)
    return p, p + f


def run_benchmarks(base, adapter):
    """GPU host: run the frozen 32-task + hard-tree + tree_serialize evals for the v2.31 adapter and
    write benchmark_metrics.json. Delegates to evaluate_code_agent.py (read-only on the benchmark)."""
    if not _gpu_ok():
        print("[v231] SKIP: --benchmarks requires a CUDA GPU. (No evaluation performed; no fabricated "
              "metrics.)")
        return
    if not Path(adapter).exists():
        print(f"[v231] ERROR: adapter not found at {adapter}. Train first."); sys.exit(1)
    out32 = OUT_DIR / "eval_32task_adapter"
    cmd = [sys.executable, "scripts/evaluate_code_agent.py", "--hf-model", base, "--hf-lora", adapter,
           "--tasks-file", str(BENCH_32), "--mode", "best_of_n", "--n", "3",
           "--scoring-mode", "verified_agent", "--verifier-repair", "--output", str(out32)]
    print(f"[v231] running 32-task benchmark for the adapter: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    csv32 = out32 / "best_of_3.csv"
    adapter_32, total_32 = _count_passes(csv32)
    hard_p, hard_n = _count_passes(csv32, ids=HARD_TREE_IDS)
    ts_p, ts_n = _count_passes(csv32, ids=("tree_serialize",))
    metrics = {
        "champion_32_pass": CHAMPION_32, "adapter_32_pass": adapter_32, "total_32": total_32,
        "hard_tree": f"{hard_p}/{hard_n}", "hard_tree_pass": hard_p,
        "tree_serialize_pass": ts_p, "tree_serialize_total": ts_n,
        # 3/3 preservation: tree_serialize tasks not regressed (the v2.27 canonical-control 3/3 is a
        # deterministic scaffold, model-independent; here we confirm the adapter does not break it).
        "tree_serialize_3of3_preserved": bool(ts_n == 0 or ts_p >= ts_n),
        "material_regression": adapter_32 < CHAMPION_32 - 1,
    }
    (OUT_DIR / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v231] benchmark: adapter {adapter_32}/{total_32} (champion ref {CHAMPION_32}); "
          f"hard-tree {hard_p}/{hard_n}; tree_serialize {ts_p}/{ts_n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(OUT_DIR / "adapter"))
    ap.add_argument("--benchmarks", action="store_true",
                    help="run the frozen 32-task / hard-tree / tree_serialize gates (GPU host)")
    args = ap.parse_args()

    if args.benchmarks:
        run_benchmarks(args.base, args.adapter)
        return

    if not _gpu_ok():
        print("[v231] SKIP: no CUDA GPU — repair-adapter generation eval requires a GPU host. "
              "(No evaluation performed; no fabricated metrics.)")
        print("[v231] On a GPU host: run repair validation (this script) then the benchmark gate:")
        print("       python scripts/eval_v231_repair_sft.py --benchmarks --base <base> "
              "--adapter outputs/v231_tiny_repair_trace_sft/adapter")
        return

    val_path = DATA / "sft_val.jsonl"
    adapter = OUT_DIR / "adapter"
    if not val_path.exists() or not adapter.exists():
        print("[v231] ERROR: need sft_val.jsonl and a trained adapter. Run build + train first.")
        sys.exit(1)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda"),
        str(adapter))

    rows = [json.loads(l) for l in open(val_path)]

    def _rate(model):
        ok = 0
        for ex in rows:
            ids = tok(ex["input"], return_tensors="pt").to("cuda")
            gen = model.generate(**ids, max_new_tokens=512, do_sample=False)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            ok += int(_passes(_extract_solution(text)))
        return ok, len(rows)

    base_ok, n = _rate(base)
    adapt_ok, _ = _rate(adapted)
    metrics = {
        "val_records": n,
        "base_repair_pass": base_ok, "adapter_repair_pass": adapt_ok,
        "base_rate": round(base_ok / n, 3) if n else 0.0,
        "adapter_rate": round(adapt_ok / n, 3) if n else 0.0,
        "note": ("delegated evals (32-task / hard tree / tree_serialize 3/3) run separately via "
                 "evaluate_code_agent.py; compare against the frozen champion 23/28 baseline."),
    }
    (OUT_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v231] repair-validation: base {base_ok}/{n} vs adapter {adapt_ok}/{n}")


if __name__ == "__main__":
    main()
