"""
scripts/train_v233_scaffold_sft.py — v2.33 scaffold-first preservation trainer (GPU-gated).

Trains a small fresh LoRA adapter on ONLY correct tool-call / execute_code scaffold trajectories
(no repair objective). The aim is to preserve execute_code / tool-use behaviour and the frozen 32-task
benchmark before any repair adaptation is reintroduced. Reuses the v2.32 weighted collator (which
strips string metadata) so the HF collator never tries to tensorise non-tensor fields.

Same stability-first constraints (small LoRA, short max length, low LR, few steps), separate output
path, champion never overwritten. GPU-gated: skips cleanly with no fabricated metrics on CPU.

Usage:
    python scripts/train_v233_scaffold_sft.py [--base <hf_id>] [--max-steps 60]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_v232_mixed_sft import WeightedDataCollator, weighted_lm_loss  # noqa: E402

DATA = ROOT / "data" / "generated" / "v233"
OUT_DIR = ROOT / "outputs" / "v233_scaffold_first_sft"   # LOCAL-ONLY, gitignored, separate path


def _precheck():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"[v233] SKIP: torch unavailable ({e})."); return None
    if not torch.cuda.is_available():
        print("[v233] SKIP: no CUDA GPU available — this CPU-only environment cannot run the scaffold "
              "SFT. Run on a GPU host. (No training performed; no fabricated metrics.)")
        return None
    try:
        import peft, transformers  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[v233] SKIP: training deps missing ({e})."); return None
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if _precheck() is None:
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)

    train_path = DATA / "scaffold_train.jsonl"
    if not train_path.exists():
        print("[v233] ERROR: run make build-v233-scaffold-dataset first."); sys.exit(1)
    assert "qwen15b_memory_300steps" not in str(OUT_DIR) and "qwen3b" not in str(OUT_DIR), \
        "refusing to write under a protected path"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    ds = load_dataset("json", data_files=str(train_path), split="train")

    def _tok(ex):
        enc = tok(ex["input"] + "\n" + ex["output"] + tok.eos_token, truncation=True,
                  max_length=args.max_length, padding="max_length")
        enc["loss_weight"] = float(ex.get("loss_weight", 1.0))
        return enc

    ds = ds.map(_tok, remove_columns=ds.column_names)

    class ScaffoldTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            loss_weight = inputs.pop("loss_weight", None)
            outputs = model(**inputs)
            loss = weighted_lm_loss(outputs.logits, inputs["labels"], loss_weight)
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=args.lr, max_steps=args.max_steps,
        logging_steps=5, save_steps=args.max_steps, bf16=True, report_to=[],
        warmup_ratio=0.1, lr_scheduler_type="cosine", remove_unused_columns=False)
    trainer = ScaffoldTrainer(model=model, args=targs, train_dataset=ds,
                              data_collator=WeightedDataCollator(DataCollatorForLanguageModeling(tok, mlm=False)))
    result = trainer.train()
    model.save_pretrained(str(OUT_DIR / "adapter"))
    tok.save_pretrained(str(OUT_DIR / "adapter"))
    loss_trend = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    (OUT_DIR / "training_metrics.json").write_text(json.dumps({
        "base": args.base, "max_steps": args.max_steps, "lr": args.lr, "objective": "scaffold_only",
        "train_records": len(ds), "final_loss": getattr(result, "training_loss", None),
        "loss_trend": loss_trend, "adapter_path": str(OUT_DIR / "adapter"),
    }, indent=2))
    print(f"[v233] trained scaffold adapter -> {OUT_DIR/'adapter'} (loss_trend={loss_trend})")


if __name__ == "__main__":
    main()
