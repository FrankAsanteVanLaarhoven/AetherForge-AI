"""
scripts/finetune_code_agent.py
Fine-tune a distilled AetherForge checkpoint on code + tool-use data.

Starts from the distilled 128M (Qwen tokenizer) or any config.json checkpoint,
trains on data/code_agent_data.jsonl with a response-masked CE objective
(same masking as distillation: prompt tokens → -100).

Saves to outputs/aetherforge_code_agent/

Usage:
    # Generate data first:
    conda run -n ml-torch python scripts/generate_code_data.py

    # Fine-tune distilled 128M:
    conda run -n ml-torch python scripts/finetune_code_agent.py \
        --checkpoint outputs/aetherforge_distill_5k/final/model.pt \
        --config     outputs/aetherforge_distill_5k/final/config.json \
        --data       data/code_agent_data.jsonl \
        --steps      3000

    # Resume:
    conda run -n ml-torch python scripts/finetune_code_agent.py \
        --checkpoint outputs/aetherforge_distill_5k/final/model.pt \
        --config     outputs/aetherforge_distill_5k/final/config.json \
        --resume     outputs/aetherforge_code_agent/checkpoint-1500 \
        --steps      5000
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(data_path: str, tokenizer, max_length: int = 512) -> list[dict]:
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item  = json.loads(line)
            instr = item.get("instruction", "").strip()
            resp  = item.get("response", "").strip()
            if not instr or not resp:
                continue

            full_text = f"{instr}\n\n{resp}"
            ids = tokenizer.encode(full_text, add_special_tokens=True,
                                   max_length=max_length + 1, truncation=True)
            if len(ids) < 8:
                continue

            # Mask prompt tokens (-100 so they don't contribute to loss)
            prompt_ids  = tokenizer.encode(instr + "\n\n", add_special_tokens=True)
            prompt_len  = min(len(prompt_ids), len(ids) - 1)
            labels = [-100] * prompt_len + ids[prompt_len:]

            ids    = ids[:max_length]
            labels = labels[:max_length]
            rows.append({"ids": ids, "labels": labels})
    return rows


def collate(batch: list[dict], pad_id: int = 0):
    max_len = max(len(x["ids"]) for x in batch)
    ids_t   = torch.zeros(len(batch), max_len, dtype=torch.long)
    lab_t   = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["ids"])
        ids_t[i, :n] = torch.tensor(x["ids"])
        lab_t[i, :n] = torch.tensor(x["labels"])
    return ids_t, lab_t


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, step, loss, out_dir: Path):
    ckpt_dir = out_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save({
        "step": step, "loss": loss,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, ckpt_dir / "state.pt")


def load_checkpoint(path: str, model, optimizer, scheduler) -> int:
    state = torch.load(Path(path) / "state.pt", weights_only=False)
    model.load_state_dict(torch.load(Path(path) / "model.pt",
                                     map_location=DEVICE, weights_only=True))
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    print(f"Resumed from step {state['step']}  (loss {state['loss']:.4f})")
    return state["step"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(
    checkpoint:            str,
    config_path:           str,
    data_path:             str,
    resume:                str   = None,
    output:                str   = "outputs/aetherforge_code_agent",
    steps:                 int   = 3000,
    lr:                    float = 2e-5,
    batch_size:            int   = 4,
    grad_accum:            int   = 4,
    max_length:            int   = 512,
    log_every:             int   = 50,
    save_every:            int   = 500,
    use_8bit_adam:         bool  = False,
    gradient_checkpointing: bool = False,
    use_amp:               bool  = False,
    warmup_steps:          int   = 100,
):
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model + tokenizer ──────────────────────────────────────────────
    with open(config_path) as f:
        cfg = json.load(f)
    model = AetherForge(**cfg).to(DEVICE)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=False
    )

    # Load pretrained/distilled weights as starting point
    base_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(base_state)
    if use_amp and DEVICE == "cuda":
        # Store weights and run compute in bf16: halves weight+gradient+activation memory.
        # 8-bit Adam then operates on bf16 params — total VRAM ≈ 2× model bf16 size.
        model = model.to(dtype=torch.bfloat16)
        print("Precision: bf16 (weights, activations, gradients)")
    model = model.to(DEVICE)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: ON")
    amp_ctx = None   # weights already in bf16; no autocast needed
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nAetherForge [{cfg.get('d_model', '?')}d / {n_params/1e6:.1f}M params] on {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────
    print(f"Loading data from {data_path} ...")
    data = load_data(data_path, tokenizer, max_length=max_length)
    random.shuffle(data)
    print(f"  {len(data)} examples (max_length={max_length})")

    # ── Optimizer ─────────────────────────────────────────────────────
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr,
                weight_decay=0.01, betas=(0.9, 0.95),
            )
            print("Optimizer: AdamW8bit")
        except ImportError:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr,
                weight_decay=0.01, betas=(0.9, 0.95),
            )
            print("Optimizer: AdamW (fp32, bitsandbytes not found)")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=0.01, betas=(0.9, 0.95),
        )
        print("Optimizer: AdamW (fp32)")

    # Cosine LR schedule with linear warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer, scheduler)

    # ── CSV log ───────────────────────────────────────────────────────
    log_path  = out_dir / "finetune_log.csv"
    log_exists = log_path.exists() and resume
    log_f  = open(log_path, "a" if log_exists else "w", newline="")
    writer = csv.writer(log_f)
    if not log_exists:
        writer.writerow(["step", "loss", "lr", "tok_per_sec", "elapsed_s"])

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    accum_loss  = 0.0
    accum_steps = 0
    t0 = t_start = time.time()
    data_idx = 0

    def next_batch():
        nonlocal data_idx
        batch = data[data_idx % len(data): (data_idx % len(data)) + batch_size]
        if len(batch) < batch_size:
            remainder = batch_size - len(batch)
            batch = batch + data[:remainder]
        data_idx += batch_size
        return batch

    print(f"\nFine-tuning for {steps} steps (batch={batch_size}, grad_accum={grad_accum})\n")

    for step in range(start_step + 1, steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            batch = next_batch()
            ids_t, lab_t = collate(batch)
            ids_t = ids_t.to(DEVICE)
            lab_t = lab_t.to(DEVICE)

            logits = model(ids_t)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),   # fp32 loss for stability
                lab_t.view(-1),
                ignore_index=-100,
            ) / grad_accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        accum_loss  += step_loss
        accum_steps += 1

        if step % log_every == 0:
            elapsed   = time.time() - t0
            n_tokens  = batch_size * grad_accum * log_every * max_length
            tok_per_s = n_tokens / elapsed
            disp_loss = accum_loss / accum_steps
            cur_lr    = scheduler.get_last_lr()[0]
            total_el  = time.time() - t_start

            print(f"step {step:>6d}/{steps}  loss={disp_loss:.4f}  "
                  f"lr={cur_lr:.2e}  {tok_per_s:.0f} tok/s  [{total_el:.0f}s]")
            writer.writerow([step, round(disp_loss, 5),
                             round(cur_lr, 8), round(tok_per_s, 1), round(total_el, 1)])
            log_f.flush()
            accum_loss  = 0.0
            accum_steps = 0
            t0 = time.time()

        if step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, step_loss, out_dir)
            print(f"  Checkpoint saved → {out_dir}/checkpoint-{step}/")

    # ── Final save ────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")
    import shutil
    shutil.copy(config_path, final_dir / "config.json")
    print(f"\nDone! Final model → {final_dir}/model.pt")
    log_f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/aetherforge_distill_5k/final/model.pt",
                   help="Starting model weights (.pt)")
    p.add_argument("--config",     default="outputs/aetherforge_distill_5k/final/config.json",
                   help="Model config JSON")
    p.add_argument("--data",       default="data/code_agent_data.jsonl",
                   help="Training data JSONL")
    p.add_argument("--resume",     default=None,
                   help="Resume from checkpoint directory")
    p.add_argument("--output",     default="outputs/aetherforge_code_agent")
    p.add_argument("--steps",      type=int,   default=3000)
    p.add_argument("--lr",         type=float, default=2e-5,
                   help="Learning rate (lower than pretraining — we're fine-tuning)")
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch × accum)")
    p.add_argument("--max-length", type=int,   default=512,
                   help="Max sequence length (tokens)")
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--save-every", type=int,   default=500)
    p.add_argument("--8bit-adam",  dest="use_8bit_adam", action="store_true")
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true",
                   help="Enable gradient checkpointing — halves activation VRAM (~30% slower)")
    p.add_argument("--amp", dest="use_amp", action="store_true",
                   help="bf16 autocast — halves activation memory on A100/RTX GPUs")
    p.add_argument("--warmup-steps", type=int, default=100)
    args = p.parse_args()

    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        print("Run: conda run -n ml-torch python scripts/generate_code_data.py")
        sys.exit(1)

    finetune(
        checkpoint             = args.checkpoint,
        config_path            = args.config,
        data_path              = args.data,
        resume                 = args.resume,
        output                 = args.output,
        steps                  = args.steps,
        lr                     = args.lr,
        batch_size             = args.batch_size,
        grad_accum             = args.grad_accum,
        max_length             = args.max_length,
        log_every              = args.log_every,
        save_every             = args.save_every,
        use_8bit_adam          = args.use_8bit_adam,
        gradient_checkpointing = args.gradient_checkpointing,
        use_amp                = args.use_amp,
        warmup_steps           = args.warmup_steps,
    )


if __name__ == "__main__":
    main()
