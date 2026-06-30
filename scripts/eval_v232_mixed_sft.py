"""
scripts/eval_v232_mixed_sft.py — v2.32 mixed-adapter evaluation harness (GPU-gated).

Evaluates the v2.32 split-loss adapter on three things, on a GPU host:
  - repair validation (corrected solution executes) — base vs adapter,
  - tool-use preservation (output keeps the execute_code scaffold AND the final solution passes),
  - the frozen 32-task / hard-tree / tree_serialize benchmark gate (--benchmarks; delegates to
    evaluate_code_agent.py with the v2.32 adapter).

GPU-gated: with no CUDA device it prints a SKIP notice and exits 0 (no fabricated metrics).

Usage:
    python scripts/eval_v232_mixed_sft.py [--base <id>]
    python scripts/eval_v232_mixed_sft.py --benchmarks [--base <id>]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_v231_repair_sft import (  # noqa: E402  (reuse helpers)
    BENCH_32, CHAMPION_32, HARD_TREE_IDS, _count_passes, _extract_solution, _gpu_ok, _passes,
)

DATA = ROOT / "data" / "generated" / "v232"
OUT_DIR = ROOT / "outputs" / "v232_tool_use_preservation_sft"


def run_benchmarks(base, adapter):
    if not _gpu_ok():
        print("[v232] SKIP: --benchmarks requires a CUDA GPU. (No evaluation; no fabricated metrics.)")
        return
    if not Path(adapter).exists():
        print(f"[v232] ERROR: adapter not found at {adapter}."); sys.exit(1)
    out32 = OUT_DIR / "eval_32task_adapter"
    cmd = [sys.executable, "scripts/evaluate_code_agent.py", "--hf-model", base, "--hf-lora", adapter,
           "--tasks-file", str(BENCH_32), "--mode", "best_of_n", "--n", "3",
           "--scoring-mode", "verified_agent", "--verifier-repair", "--output", str(out32)]
    print(f"[v232] 32-task benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    csv32 = out32 / "best_of_3.csv"
    adapter_32, total_32 = _count_passes(csv32)
    hard_p, hard_n = _count_passes(csv32, ids=HARD_TREE_IDS)
    ts_p, ts_n = _count_passes(csv32, ids=("tree_serialize",))
    (OUT_DIR / "benchmark_metrics.json").write_text(json.dumps({
        "champion_32_pass": CHAMPION_32, "adapter_32_pass": adapter_32, "total_32": total_32,
        "hard_tree": f"{hard_p}/{hard_n}", "tree_serialize_pass": ts_p, "tree_serialize_total": ts_n,
        "tree_serialize_3of3_preserved": bool(ts_n == 0 or ts_p >= ts_n),
        "material_regression": adapter_32 < CHAMPION_32 - 1,
    }, indent=2))
    print(f"[v232] benchmark: adapter {adapter_32}/{total_32} (champion {CHAMPION_32}); "
          f"hard-tree {hard_p}/{hard_n}; tree_serialize {ts_p}/{ts_n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(OUT_DIR / "adapter"))
    ap.add_argument("--benchmarks", action="store_true")
    args = ap.parse_args()

    if args.benchmarks:
        run_benchmarks(args.base, args.adapter)
        return

    if not _gpu_ok():
        print("[v232] SKIP: no CUDA GPU — repair + tool-use-preservation generation eval requires a GPU "
              "host. (No evaluation; no fabricated metrics.)")
        print("[v232] On a GPU host, then run the benchmark gate:")
        print("       python scripts/eval_v232_mixed_sft.py --benchmarks --base <base>")
        return

    val_path = DATA / "sft_val.jsonl"
    if not val_path.exists() or not Path(args.adapter).exists():
        print("[v232] ERROR: need sft_val.jsonl and a trained adapter."); sys.exit(1)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda"),
        args.adapter)
    rows = [json.loads(l) for l in open(val_path)]
    repair_rows = [r for r in rows if r.get("objective") == "repair"]
    pres_rows = [r for r in rows if r.get("objective") == "tool_use_preservation"]

    def _gen(model, text):
        ids = tok(text, return_tensors="pt").to("cuda")
        g = model.generate(**ids, max_new_tokens=512, do_sample=False)
        return tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _repair_rate(model):
        return sum(_passes(_extract_solution(_gen(model, r["input"]))) for r in repair_rows), len(repair_rows)

    def _preservation_rate(model):
        ok = 0
        for r in pres_rows:
            t = _gen(model, r["input"])
            ok += int(("execute_code" in t) and _passes(_extract_solution(t) or t))
        return ok, len(pres_rows)

    base_r, n_r = _repair_rate(base)
    adapt_r, _ = _repair_rate(adapted)
    adapt_p, n_p = _preservation_rate(adapted)
    (OUT_DIR / "eval_metrics.json").write_text(json.dumps({
        "val_records": len(rows), "val_repair": n_r, "val_preservation": n_p,
        "base_repair_pass": base_r, "adapter_repair_pass": adapt_r,
        "adapter_preservation_pass": adapt_p,
        "base_rate": round(base_r / n_r, 3) if n_r else 0.0,
        "adapter_rate": round(adapt_r / n_r, 3) if n_r else 0.0,
        "preservation_rate": round(adapt_p / n_p, 3) if n_p else 0.0,
    }, indent=2))
    print(f"[v232] repair: base {base_r}/{n_r} vs adapter {adapt_r}/{n_r}; "
          f"tool-use preservation: adapter {adapt_p}/{n_p}")


if __name__ == "__main__":
    main()
