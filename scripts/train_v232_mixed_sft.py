"""
scripts/train_v232_mixed_sft.py — v2.32 split-loss mixed SFT trainer (GPU-gated).

Trains a small fresh LoRA adapter on the v2.32 mixed dataset (repair objective + tool-use
preservation objective) with a SPLIT LOSS: each example carries a `loss_weight` (repair=1.0,
preservation=configurable) and the trainer scales the per-example language-model loss accordingly.
Same stability-first constraints as v2.31 (small adapter, short max length, low LR, few steps),
separate output path, champion never overwritten.

GPU-gated: with no CUDA device (or missing deps) it prints a SKIP notice and exits 0 without training
and without fabricating metrics. Run on a GPU host to produce outputs/v232_tool_use_preservation_sft/.

Usage:
    python scripts/train_v232_mixed_sft.py [--base <hf_id>] [--max-steps 80]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "generated" / "v232"
OUT_DIR = ROOT / "outputs" / "v232_tool_use_preservation_sft"   # LOCAL-ONLY, gitignored, separate path


def _precheck():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"[v232] SKIP: torch unavailable ({e})."); return None
    if not torch.cuda.is_available():
        print("[v232] SKIP: no CUDA GPU available — this CPU-only environment cannot run the split-loss "
              "SFT. Run on a GPU host. (No training performed; no fabricated metrics.)")
        return None
    try:
        import peft, transformers  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[v232] SKIP: training deps missing ({e})."); return None
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--max-steps", type=int, default=80)
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

    train_path = DATA / "sft_train.jsonl"
    if not train_path.exists():
        print("[v232] ERROR: run make build-v232-mixed-dataset first."); sys.exit(1)
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
        enc["objective"] = ex.get("objective", "repair")
        return enc

    ds = ds.map(_tok, remove_columns=ds.column_names)

    class SplitLossTrainer(Trainer):
        """Scales each example's LM loss by its objective loss_weight (repair vs preservation)."""
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            weights = inputs.pop("loss_weight", None)
            inputs.pop("objective", None)
            out = model(**inputs)
            loss = out.loss
            if weights is not None:
                loss = loss * weights.to(loss.device).float().mean()
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=args.lr, max_steps=args.max_steps,
        logging_steps=5, save_steps=args.max_steps, bf16=True, report_to=[],
        warmup_ratio=0.1, lr_scheduler_type="cosine", remove_unused_columns=False)
    trainer = SplitLossTrainer(model=model, args=targs, train_dataset=ds,
                               data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    result = trainer.train()
    model.save_pretrained(str(OUT_DIR / "adapter"))
    tok.save_pretrained(str(OUT_DIR / "adapter"))
    agg = json.loads((DATA / "mixed_aggregate.json").read_text())
    loss_trend = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    (OUT_DIR / "training_metrics.json").write_text(json.dumps({
        "base": args.base, "max_steps": args.max_steps, "lr": args.lr,
        "train_records": len(ds), "mix": {"repair": agg["repair_examples"],
                                          "preservation": agg["tool_use_preservation_examples"]},
        "preservation_loss_weight": agg["preservation_loss_weight"],
        "final_loss": getattr(result, "training_loss", None), "loss_trend": loss_trend,
        "adapter_path": str(OUT_DIR / "adapter"),
    }, indent=2))
    print(f"[v232] trained split-loss adapter -> {OUT_DIR/'adapter'} (loss_trend={loss_trend})")


if __name__ == "__main__":
    main()
