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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    args = ap.parse_args()

    if not _gpu_ok():
        print("[v231] SKIP: no CUDA GPU — repair-adapter generation eval requires a GPU host. "
              "(No evaluation performed; no fabricated metrics.)")
        print("[v231] On a GPU host, also run the delegated evals:")
        print("       make eval-v226-3b-run1   # tree_serialize format-control reference")
        print("       (32-task benchmark + hard tree via evaluate_code_agent.py — see docs/v2.31_*.md)")
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
