"""
scripts/eval_v233_scaffold_sft.py — v2.33 scaffold-adapter evaluation harness (GPU-gated).

Evaluates the scaffold-first adapter on what v2.33 actually cares about — tool-use preservation and
the frozen 32-task benchmark — NOT repair. On a GPU host:
  - tool-use preservation validation (output keeps the execute_code scaffold AND the final passes),
  - the frozen 32-task / hard-tree / tree_serialize benchmark (--benchmarks; delegates to
    evaluate_code_agent.py with the v2.33 adapter), with a failure-reason breakdown (esp. no_tool_call),
    tool_call_rate, and execute_code_rate.

Repair is reported only as an OPTIONAL diagnostic, never as a promotion gate.

GPU-gated: with no CUDA device it prints a SKIP notice and exits 0 (no fabricated metrics).

Usage:
    python scripts/eval_v233_scaffold_sft.py            # tool-use preservation validation
    python scripts/eval_v233_scaffold_sft.py --benchmarks
"""

import argparse
import csv as _csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_csv.field_size_limit(10_000_000)

from scripts.eval_v231_repair_sft import (  # noqa: E402  (reuse helpers)
    BENCH_32, CHAMPION_32, HARD_TREE_IDS, _count_passes, _extract_solution, _gpu_ok, _passes,
)

DATA = ROOT / "data" / "generated" / "v233"
OUT_DIR = ROOT / "outputs" / "v233_scaffold_first_sft"


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _scaffold_stats(csv_path):
    """tool_call_rate, execute_code_rate, no_tool_call count, and failure-reason breakdown."""
    n = tool_calls = exec_code = no_tool = 0
    reasons = Counter()
    for row in _csv.DictReader(open(csv_path)):
        n += 1
        made_call = bool((row.get("first_tool_call") or "").strip()) or \
            (str(row.get("tool_calls", "0")).strip() not in ("", "0"))
        used_exec = _truthy(row.get("used_execute_code")) or \
            ((row.get("first_tool_call") or "").strip() == "execute_code")
        tool_calls += int(made_call)
        exec_code += int(used_exec)
        if not made_call:
            no_tool += 1
        if not _truthy(row.get("passed")):
            reasons[(row.get("failure_reason") or ("no_tool_call" if not made_call else "other")).strip()] += 1
    return {
        "n": n, "tool_call_rate": round(tool_calls / n, 3) if n else 0.0,
        "execute_code_rate": round(exec_code / n, 3) if n else 0.0,
        "no_tool_call": no_tool,
        "no_tool_call_dominant": bool(reasons) and reasons.most_common(1)[0][0] == "no_tool_call",
        "failure_reasons": dict(reasons),
    }


def run_benchmarks(base, adapter):
    if not _gpu_ok():
        print("[v233] SKIP: --benchmarks requires a CUDA GPU. (No evaluation; no fabricated metrics.)")
        return
    if not Path(adapter).exists():
        print(f"[v233] ERROR: adapter not found at {adapter}."); sys.exit(1)
    out32 = OUT_DIR / "eval_32task_adapter"
    cmd = [sys.executable, "scripts/evaluate_code_agent.py", "--hf-model", base, "--hf-lora", adapter,
           "--tasks-file", str(BENCH_32), "--mode", "best_of_n", "--n", "3",
           "--scoring-mode", "verified_agent", "--verifier-repair", "--output", str(out32)]
    print(f"[v233] 32-task benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    csv32 = out32 / "best_of_3.csv"
    adapter_32, total_32 = _count_passes(csv32)
    hard_p, hard_n = _count_passes(csv32, ids=HARD_TREE_IDS)
    ts_p, ts_n = _count_passes(csv32, ids=("tree_serialize",))
    stats = _scaffold_stats(csv32)
    metrics = {
        "champion_32_pass": CHAMPION_32, "adapter_32_pass": adapter_32, "total_32": total_32,
        "hard_tree": f"{hard_p}/{hard_n}", "tree_serialize_pass": ts_p, "tree_serialize_total": ts_n,
        "tree_serialize_3of3_preserved": bool(ts_n == 0 or ts_p >= ts_n),
        "material_regression": adapter_32 < CHAMPION_32 - 1,
        **stats,
    }
    (OUT_DIR / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[v233] benchmark: adapter {adapter_32}/{total_32} (champion {CHAMPION_32}); "
          f"tool_call_rate {stats['tool_call_rate']}; no_tool_call {stats['no_tool_call']} "
          f"(dominant={stats['no_tool_call_dominant']}); tree_serialize {ts_p}/{ts_n}")


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
        print("[v233] SKIP: no CUDA GPU — tool-use preservation generation eval requires a GPU host. "
              "(No evaluation; no fabricated metrics.)")
        print("[v233] On a GPU host, then run the benchmark gate:")
        print("       python scripts/eval_v233_scaffold_sft.py --benchmarks --base <base>")
        return

    val_path = DATA / "scaffold_val.jsonl"
    if not val_path.exists() or not Path(args.adapter).exists():
        print("[v233] ERROR: need scaffold_val.jsonl and a trained adapter."); sys.exit(1)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda"),
        args.adapter)
    rows = [json.loads(l) for l in open(val_path)]
    ok = tool_call = 0
    for r in rows:
        ids = tok(r["input"], return_tensors="pt").to("cuda")
        g = adapted.generate(**ids, max_new_tokens=512, do_sample=False)
        text = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        has_tool = "execute_code" in text
        tool_call += int(has_tool)
        ok += int(has_tool and _passes(_extract_solution(text) or text))
    (OUT_DIR / "eval_metrics.json").write_text(json.dumps({
        "val_records": len(rows), "preservation_pass": ok,
        "tool_call_rate": round(tool_call / len(rows), 3) if rows else 0.0,
        "preservation_rate": round(ok / len(rows), 3) if rows else 0.0,
    }, indent=2))
    print(f"[v233] tool-use preservation: {ok}/{len(rows)} (tool_call_rate "
          f"{round(tool_call / len(rows), 3) if rows else 0.0})")


if __name__ == "__main__":
    main()
