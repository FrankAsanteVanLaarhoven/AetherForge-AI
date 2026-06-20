"""
Fine-tune Qwen2.5-VL-7B-Instruct with 4-bit NF4 + LoRA on RTX 4080 16GB.

Modes:
  text      — text-only instruction fine-tuning (no images needed)
  multimodal — text + image fine-tuning (requires image paths in dataset)

Usage:
    # Text-only (works immediately, no images needed)
    conda run -n ml-torch python scripts/finetune_qwen25_vl.py --mode text

    # Multimodal (images + text)
    conda run -n ml-torch python scripts/finetune_qwen25_vl.py \
        --mode multimodal --data data/vl_dataset.jsonl

    # Quick smoke-test (25 steps, tiny batch)
    conda run -n ml-torch python scripts/finetune_qwen25_vl.py --mode text --test-run

    # With W&B logging
    conda run -n ml-torch python scripts/finetune_qwen25_vl.py --mode text --wandb
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import torch
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import LoraConfig, get_peft_model, TaskType
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = Path("./outputs/qwen25_vl_lora")

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Single-turn instruction dataset from a JSONL file.

    Each line: {"instruction": "...", "response": "..."}
    Falls back to a small built-in sample if no file is provided.

    Labels are masked for the prompt (user turn), so loss is computed only
    on the assistant response — this prevents the model from memorising
    fixed question phrasing instead of learning response quality.
    """

    BUILTIN = [
        {"instruction": "What is the Transformer architecture?",
         "response": "The Transformer uses self-attention to model dependencies "
                     "between all positions in a sequence simultaneously, replacing "
                     "recurrent networks with parallelisable attention layers."},
        {"instruction": "Explain sparse Mixture of Experts in one paragraph.",
         "response": "Sparse MoE routes each token to a small subset (top-k) of "
                     "specialised feed-forward networks called experts. A learned router "
                     "assigns weights; only the selected experts execute, so total "
                     "compute per token stays low while model capacity scales with the "
                     "number of experts."},
        {"instruction": "What is 4-bit NF4 quantisation?",
         "response": "NF4 (NormalFloat4) maps weights to a 4-bit grid whose levels are "
                     "optimally spaced for normally distributed weights. Double "
                     "quantisation then quantises the quantisation constants themselves, "
                     "saving an additional ~0.4 bits/parameter with minimal accuracy loss."},
        {"instruction": "What is LoRA?",
         "response": "LoRA (Low-Rank Adaptation) inserts trainable low-rank matrices "
                     "A and B into each attention projection so that only r×(d_in+d_out) "
                     "parameters are updated instead of d_in×d_out, reducing fine-tuning "
                     "VRAM by up to 10× with minimal performance loss."},
        {"instruction": "Describe a hospital navigation task for a mobile robot.",
         "response": "The robot must localise itself in a dynamic environment populated "
                     "by staff, patients, and equipment. It plans collision-free paths "
                     "to target rooms, adjusts speed near pedestrians, and communicates "
                     "its intent via audio and lighting cues while complying with "
                     "hygiene and safety protocols."},
        {"instruction": "What is Flash Attention and why does it matter?",
         "response": "Flash Attention rewrites the attention kernel to tile Q/K/V in "
                     "SRAM, avoiding materialising the full N×N attention matrix in HBM. "
                     "This reduces memory from O(N²) to O(N) and achieves 2–4× wall-clock "
                     "speedup for long sequences without changing the mathematical output."},
    ] * 40

    def __init__(self, processor, path: Optional[str] = None,
                 max_length: int = 256):
        self.processor  = processor
        self.max_length = max_length
        if path and Path(path).exists():
            with open(path) as f:
                self.data = [json.loads(l) for l in f if l.strip()]
            print(f"Loaded {len(self.data)} samples from {path}")
        else:
            self.data = self.BUILTIN
            print(f"Using built-in dataset ({len(self.data)} samples)")

    def __len__(self):
        return len(self.data)

    def _prompt_length(self, instruction: str) -> int:
        """Tokenized length of the user prompt (no response) — used for label masking."""
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = self.processor(text=prompt, return_tensors="pt")
        return enc["input_ids"].shape[1]

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "user",      "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        # Mask padding
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # Mask user prompt — compute loss on response only
        prompt_len = min(self._prompt_length(item["instruction"]), self.max_length)
        labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class MultimodalDataset(Dataset):
    """Vision-language dataset from JSONL.

    Each line: {"instruction": "...", "response": "...", "image": "/path/to/img.jpg"}

    Images are loaded as PIL and passed to the Qwen2.5-VL processor correctly.
    Loss is masked on the user prompt — only the assistant response contributes.
    """

    def __init__(self, processor, path: str, max_length: int = 256):
        self.processor  = processor
        self.max_length = max_length
        if not Path(path).exists():
            print(f"ERROR: dataset not found at {path}")
            sys.exit(1)
        with open(path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        # Filter to rows with a valid image path
        self.data = [d for d in self.data if d.get("image") and Path(d["image"]).exists()]
        print(f"Loaded {len(self.data)} multimodal samples from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        item  = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item["instruction"]},
            ]},
            {"role": "assistant", "content": item["response"]},
        ]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(
            text=full_text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # Prompt-only length (image token + instruction, no response)
        prompt_msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": item["instruction"]},
        ]}]
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        prompt_enc  = self.processor(text=prompt_text, images=[image],
                                     return_tensors="pt")
        prompt_len  = min(prompt_enc["input_ids"].shape[1], self.max_length)
        labels[:prompt_len] = -100

        out = {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        # pixel_values: processor returns [N_patches, C] (no batch dim already)
        # image_grid_thw: [n_images, 3] — required by Qwen2.5-VL vision position encoding
        if "pixel_values" in enc:
            out["pixel_values"]    = enc["pixel_values"]
        if "image_grid_thw" in enc:
            out["image_grid_thw"] = enc["image_grid_thw"]
        return out


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss curve (saved during training + at end)
# ---------------------------------------------------------------------------

def save_loss_curve(log_path: Path, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(log_path)
        if df.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0D1117")
        for ax in axes:
            ax.set_facecolor("#161B22")
            ax.tick_params(colors="#E6EDF3")
            ax.xaxis.label.set_color("#E6EDF3")
            ax.yaxis.label.set_color("#E6EDF3")
            ax.title.set_color("#E6EDF3")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363D")

        # Loss
        axes[0].plot(df["step"], df["loss"], color="#58A6FF", linewidth=1.2,
                     alpha=0.6, label="raw")
        if len(df) >= 5:
            ma = df["loss"].rolling(5, min_periods=1).mean()
            axes[0].plot(df["step"], ma, color="#E3B341", linewidth=2,
                         label="MA-5")
        best_row = df.loc[df["loss"].idxmin()]
        axes[0].scatter([best_row["step"]], [best_row["loss"]],
                        marker="D", color="#3FB950", s=60, zorder=5, label="best")
        axes[0].set(xlabel="Step", ylabel="Loss", title="Training Loss")
        axes[0].legend(framealpha=0.3)
        axes[0].grid(alpha=0.2, color="#30363D")

        # LR
        axes[1].fill_between(df["step"], df["lr"], alpha=0.4, color="#8B949E")
        axes[1].plot(df["step"], df["lr"], color="#8B949E", linewidth=1.5)
        axes[1].set(xlabel="Step", ylabel="LR", title="Learning Rate")
        axes[1].grid(alpha=0.2, color="#30363D")

        # VRAM
        axes[2].plot(df["step"], df["vram_gb"], color="#3FB950", linewidth=1.5)
        axes[2].axhline(16.0, color="#F85149", linewidth=1, linestyle="--",
                        alpha=0.6, label="16 GB budget")
        axes[2].set(xlabel="Step", ylabel="VRAM (GB)", title="GPU Memory")
        axes[2].legend(framealpha=0.3)
        axes[2].grid(alpha=0.2, color="#30363D")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Loss curve saved to {out_path}")
    except ImportError:
        pass   # pandas/matplotlib not installed — silently skip


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_processor(lora_r: int = 16):
    global LORA_CONFIG
    LORA_CONFIG = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    print(f"Loading {MODEL_ID} in 4-bit NF4 ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, processor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    mode:           str   = "text",
    data_path:      Optional[str] = None,
    batch_size:     int   = 1,
    grad_accum:     int   = 8,
    max_length:     int   = 256,
    lr:             float = 2e-4,
    n_epochs:       int   = 1,
    warmup_steps:   int   = 20,
    save_steps:     int   = 50,
    plot_every:     int   = 20,
    test_run:       bool  = False,
    use_wandb:      bool  = False,
    wandb_project:  str   = "aetherforge",
    lora_r:         int   = 16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(lora_r)

    if mode == "multimodal":
        if not data_path:
            print("ERROR: --data required for multimodal mode.")
            sys.exit(1)
        dataset = MultimodalDataset(processor, data_path, max_length)
    else:
        dataset = TextDataset(processor, data_path, max_length)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0,                   # 0 avoids PIL serialisation issues in subprocess
        pin_memory=(device == "cuda"),
    )

    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    total_steps = (len(loader) * n_epochs) // grad_accum
    if test_run:
        total_steps = min(total_steps, 25)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler("cuda") if device == "cuda" else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV log — compatible with Training_Dashboard.ipynb
    log_path   = OUTPUT_DIR / "training_log.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "loss", "lr", "vram_gb",
                         "tokens_per_sec", "elapsed_sec"])

    if use_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, config={
                "mode": mode, "batch_size": batch_size,
                "grad_accum": grad_accum, "max_length": max_length,
                "lr": lr, "lora_r": lora_r, "warmup_steps": warmup_steps,
            })
        except ImportError:
            print("wandb not installed — skipping. pip install wandb")
            use_wandb = False

    model.train()
    global_step  = 0
    accum_loss   = 0.0
    accum_tokens = 0
    t_start = time.time()
    t_step  = time.time()

    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(loader):
            if test_run and global_step >= total_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = dict(input_ids=input_ids, labels=labels,
                          attention_mask=attention_mask)

            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                # DataLoader adds a leading batch dim; squeeze it back to
                # [N_patches, C] which is what Qwen2.5-VL's vision encoder expects.
                pv = pixel_values.squeeze(0).to(device, dtype=torch.float16)
                kwargs["pixel_values"] = pv
            if image_grid_thw is not None:
                # DataLoader wraps [n_img, 3] → [1, n_img, 3]; squeeze to [n_img, 3].
                kwargs["image_grid_thw"] = image_grid_thw.squeeze(0).to(device)

            if device == "cuda":
                with autocast("cuda"):
                    outputs = model(**kwargs)
                    loss    = outputs.loss / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(**kwargs)
                loss    = outputs.loss / grad_accum
                loss.backward()

            accum_loss   += loss.item()
            accum_tokens += (labels != -100).sum().item()

            if (batch_idx + 1) % grad_accum == 0:
                if device == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                now         = time.time()
                elapsed     = now - t_start
                step_time   = now - t_step
                tok_per_sec = accum_tokens / max(step_time, 1e-6)
                remaining   = (total_steps - global_step) * step_time
                eta         = f"{int(remaining//60)}m{int(remaining%60)}s"
                vram        = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
                lr_now      = scheduler.get_last_lr()[0]
                disp_loss   = accum_loss * grad_accum   # actual step loss

                print(
                    f"epoch {epoch+1} | step {global_step:4d}/{total_steps} | "
                    f"loss {disp_loss:.4f} | lr {lr_now:.2e} | "
                    f"vram {vram:.1f}GB | {tok_per_sec:.0f} tok/s | eta {eta}"
                )

                log_writer.writerow([global_step, epoch + 1,
                                     round(disp_loss, 6), round(lr_now, 8),
                                     round(vram, 2), round(tok_per_sec, 1),
                                     round(elapsed, 1)])
                log_file.flush()

                if use_wandb:
                    import wandb
                    wandb.log({"loss": disp_loss, "lr": lr_now,
                               "vram_gb": vram, "tok_per_sec": tok_per_sec},
                              step=global_step)

                # Auto-save loss curve during training
                if global_step % plot_every == 0:
                    save_loss_curve(log_path, OUTPUT_DIR / "loss_curve.png")

                if global_step % save_steps == 0:
                    ckpt = OUTPUT_DIR / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    print(f"  saved -> {ckpt}")

                accum_loss   = 0.0
                accum_tokens = 0
                t_step       = now

    log_file.close()
    if use_wandb:
        import wandb
        wandb.finish()

    final_dir = OUTPUT_DIR / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nTraining complete. Model: {final_dir}")
    print(f"Log: {log_path}")

    save_loss_curve(log_path, OUTPUT_DIR / "loss_curve.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text",
                        help="text: instruction tuning only; multimodal: image+text JSONL")
    parser.add_argument("--data", default=None,
                        help="JSONL dataset path (uses built-in 240-sample set if omitted)")
    parser.add_argument("--output-dir", default="./outputs/qwen25_vl_lora")
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--grad-accum",   type=int,   default=8,
                        help="Gradient accumulation — effective batch = batch-size × grad-accum")
    parser.add_argument("--max-length",   type=int,   default=256,
                        help="Sequence length (256 is safe on 16GB when Isaac Sim is running)")
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--warmup-steps", type=int,   default=20,
                        help="Linear LR warmup before cosine decay")
    parser.add_argument("--save-steps",   type=int,   default=50,
                        help="Save checkpoint every N optimiser steps")
    parser.add_argument("--plot-every",   type=int,   default=20,
                        help="Re-save loss_curve.png every N steps (0 = only at end)")
    parser.add_argument("--lora-r",       type=int,   default=16,
                        help="LoRA rank (higher = more capacity, more VRAM)")
    parser.add_argument("--wandb",         dest="use_wandb", action="store_true",
                        help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="aetherforge")
    parser.add_argument("--test-run",     action="store_true",
                        help="Run 25 steps — fast pipeline verification")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)

    train(
        mode          = args.mode,
        data_path     = args.data,
        batch_size    = args.batch_size,
        grad_accum    = args.grad_accum,
        max_length    = args.max_length,
        lr            = args.lr,
        n_epochs      = args.epochs,
        warmup_steps  = args.warmup_steps,
        save_steps    = args.save_steps,
        plot_every    = args.plot_every,
        lora_r        = args.lora_r,
        test_run      = args.test_run,
        use_wandb     = args.use_wandb,
        wandb_project = args.wandb_project,
    )


if __name__ == "__main__":
    main()
