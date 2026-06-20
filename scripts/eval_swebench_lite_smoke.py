#!/usr/bin/env python3
"""SWE-bench Lite smoke-test scaffold for AetherForge-AI.

IMPORTANT — what this script does and does NOT do:
  - DOES: load SWE-bench Lite tasks, run the AetherForge agent, collect patch output
  - DOES: write predictions in the official SWE-bench format
  - DOES NOT: run the official Docker-based harness (required for verified scores)
  - DOES NOT: claim any issue is resolved (only the harness can do that)

Official evaluation: https://github.com/princeton-nlp/SWE-bench

TODO — true SWE-bench support requires repo-level agent tools not yet implemented:
  - read_file(path): read a source file from the cloned repository
  - search_repo(pattern): grep/search across the repository
  - edit_file(path, old, new): patch a specific file in place
  - run_tests(test_ids): run the real test suite and return pass/fail
  - generate_diff(): produce a unified git diff of all edits

Until these tools exist, this script generates placeholder patches using only
the issue description. Expect very low (likely 0%) resolution rates on the
official harness — that is expected and acceptable at this milestone.
"""

import argparse
import json
import pathlib
import sys
import textwrap
import time

# ---------------------------------------------------------------------------
# Optional model import — graceful failure if transformers not available
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


def _load_model(model_path: str):
    """Load a local LoRA or base HF model. Returns (model, tokenizer) or (None, None)."""
    if not _TRANSFORMERS_OK:
        print("[warn] transformers/peft not available — running in stub mode (empty patches)")
        return None, None
    try:
        import os
        # Detect whether this is a LoRA adapter dir or a full model
        adapter_cfg = pathlib.Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
            print(f"[model] Loading base {base} + LoRA adapter {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print(f"[model] Loading model {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        model.eval()
        return model, tokenizer
    except Exception as exc:
        print(f"[warn] Could not load model: {exc}")
        print("[warn] Running in stub mode (empty patches)")
        return None, None


def _generate_patch(model, tokenizer, instance_id: str, repo: str,
                    problem_statement: str, max_new_tokens: int = 512) -> str:
    """Attempt to generate a unified diff patch for the given issue.

    Returns a string in unified diff format, or an empty string if stub mode.
    The patch is NOT verified — it must be submitted to the official harness.
    """
    if model is None or tokenizer is None:
        # Stub: return an empty patch (will not resolve anything in the harness,
        # but produces a syntactically valid prediction record)
        return ""

    prompt = textwrap.dedent(f"""\
        You are an expert Python developer. You must produce a minimal unified diff
        (git diff format) that fixes the issue described below.

        Repository: {repo}
        Issue: {problem_statement[:1500]}

        Output ONLY the unified diff, starting with 'diff --git'. No explanation.
        """)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Try to extract a diff block if present
    if "diff --git" in generated:
        start = generated.index("diff --git")
        return generated[start:].strip()
    return ""


def _load_swebench_lite(limit: int):
    """Load the first `limit` tasks from princeton-nlp/SWE-bench_Lite."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading princeton-nlp/SWE-bench_Lite (split=test, limit={limit}) …")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = list(ds.select(range(min(limit, len(ds)))))
    print(f"Loaded {len(tasks)} task(s)")
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite smoke-test scaffold for AetherForge-AI"
    )
    parser.add_argument(
        "--model", default="outputs/qwen15b_memory_300steps/final",
        help="Path to local HF model or LoRA adapter directory"
    )
    parser.add_argument(
        "--limit", type=int, default=3,
        help="Number of SWE-bench Lite tasks to process (default: 3 for smoke test)"
    )
    parser.add_argument(
        "--output-dir", default="outputs/swebench_lite_smoke",
        help="Directory to write predictions.jsonl"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens for patch generation"
    )
    parser.add_argument(
        "--stub-only", action="store_true",
        help="Skip model loading and generate empty-patch stubs only (for CI/testing)"
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    print()
    print("=" * 72)
    print("AetherForge-AI  —  SWE-bench Lite Smoke Evaluation")
    print("=" * 72)
    print("IMPORTANT: Patch generation only; official harness evaluation still required.")
    print("  Generated patches must be verified using the official SWE-bench harness.")
    print("  See: https://github.com/princeton-nlp/SWE-bench")
    print()
    print("Scope of this script:")
    print("  - Load SWE-bench Lite tasks from Hugging Face")
    print("  - Generate candidate patches using AetherForge (function-level model)")
    print("  - Save predictions in official SWE-bench format")
    print("  - DO NOT claim any issues are resolved")
    print()
    print("Known limitation:")
    print("  The current AetherForge agent uses execute_code (function-level tool).")
    print("  True SWE-bench resolution requires repo-level tools (read_file,")
    print("  edit_file, run_tests, generate_diff) — NOT YET IMPLEMENTED.")
    print("  Expect 0% or near-0% resolution on the official harness at this stage.")
    print()

    # Load model
    if args.stub_only:
        model, tokenizer = None, None
        model_name = f"{args.model} [stub-only]"
    else:
        model, tokenizer = _load_model(args.model)
        model_name = args.model

    # Load tasks
    tasks = _load_swebench_lite(args.limit)

    # Generate predictions
    predictions = []
    for i, task in enumerate(tasks, 1):
        instance_id = task["instance_id"]
        repo = task.get("repo", "unknown/repo")
        problem = task.get("problem_statement", "")

        print(f"\n[{i:02d}/{len(tasks)}] {instance_id}")
        print(f"  repo: {repo}")
        print(f"  issue: {problem[:120].strip()}…")

        t0 = time.time()
        patch = _generate_patch(model, tokenizer, instance_id, repo, problem,
                                 max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - t0

        status = "patch generated" if patch else "empty patch (stub)"
        print(f"  status: {status}  ({elapsed:.1f}s)")

        # Official SWE-bench prediction format
        record = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }
        predictions.append(record)

    # Write predictions
    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    print()
    print("=" * 72)
    print(f"Wrote {len(predictions)} prediction(s) → {predictions_path}")
    print()
    print("Next steps (official evaluation):")
    print("  1. Install the official SWE-bench harness:")
    print("       pip install swebench")
    print("  2. Run the official evaluator (requires Docker):")
    print(f"       python -m swebench.harness.run_evaluation \\")
    print(f"         --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"         --predictions_path {predictions_path} \\")
    print(f"         --max_workers 1 \\")
    print(f"         --run_id aetherforge_smoke")
    print()
    print("IMPORTANT: Patch generation only; official harness evaluation still required.")
    print("=" * 72)


if __name__ == "__main__":
    main()
