"""
scripts/train_v231_repair_sft.py — v2.31 tiny repair-trace SFT pilot trainer (GPU-gated).

Trains a SMALL fresh LoRA adapter on the v2.31 repair-trace SFT export (input = failed candidate +
structured verifier signal; output = repair plan + corrected solution). First objective is STABILITY,
not maximum score: small adapter, short max length, low LR, few steps, early stop, separate output
path. The frozen champion and all prior adapters are never loaded for writing or overwritten.

This script is GPU-gated: with no CUDA device (or missing train deps) it prints a SKIP notice and
exits 0 without training, so it is safe to invoke anywhere. Run it on a GPU host to produce the
adapter at outputs/v231_tiny_repair_trace_sft/ (LOCAL-ONLY, gitignored).

Usage:
    python scripts/train_v231_repair_sft.py [--base <hf_id_or_path>] [--max-steps 60]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "generated" / "v231"
OUT_DIR = ROOT / "outputs" / "v231_tiny_repair_trace_sft"   # LOCAL-ONLY, gitignored, separate path


def _precheck():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[v231] SKIP: torch unavailable ({e}).")
        return None
    import torch
    if not torch.cuda.is_available():
        print("[v231] SKIP: no CUDA GPU available — this CPU-only environment cannot run the SFT "
              "pilot. Run on a GPU host. (No training performed; no fabricated metrics.)")
        return None
    try:
        import peft, transformers, trl  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[v231] SKIP: training deps missing ({e}). pip install peft transformers trl.")
        return None
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    help="base model id/path (read-only; champion is never overwritten)")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if _precheck() is None:
        return  # exit 0: nothing trained in a non-GPU environment

    # ── GPU path (runs on a GPU host) ─────────────────────────────────────────
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling)

    train_path = DATA / "sft_train.jsonl"
    if not train_path.exists():
        print("[v231] ERROR: run make build-v231-sft-dataset first.")
        sys.exit(1)
    assert "qwen15b_memory_300steps" not in str(OUT_DIR) and "qwen3b" not in str(OUT_DIR), \
        "refusing to write under a protected path"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files=str(train_path), split="train")

    def _tok(ex):
        text = ex["input"] + "\n" + ex["output"] + tok.eos_token
        return tok(text, truncation=True, max_length=args.max_length, padding="max_length")

    ds = ds.map(_tok, remove_columns=ds.column_names)
    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=args.lr, max_steps=args.max_steps,
        logging_steps=5, save_steps=args.max_steps, bf16=True, report_to=[],
        warmup_ratio=0.1, lr_scheduler_type="cosine",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    result = trainer.train()
    model.save_pretrained(str(OUT_DIR / "adapter"))
    tok.save_pretrained(str(OUT_DIR / "adapter"))
    loss_trend = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    (OUT_DIR / "training_metrics.json").write_text(json.dumps({
        "base": args.base, "max_steps": args.max_steps, "lr": args.lr,
        "train_records": len(ds), "final_loss": getattr(result, "training_loss", None),
        "loss_trend": loss_trend, "adapter_path": str(OUT_DIR / "adapter"),
    }, indent=2))
    print(f"[v231] trained tiny repair adapter -> {OUT_DIR/'adapter'} (loss_trend={loss_trend})")


if __name__ == "__main__":
    main()
