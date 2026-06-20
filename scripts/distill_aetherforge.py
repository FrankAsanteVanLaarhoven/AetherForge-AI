"""
scripts/distill_aetherforge.py

Knowledge distillation: Qwen2.5-VL-7B-Instruct teacher → AetherForge student.

Training objective:
    L = α · T² · KL(p_teacher || p_student) + (1-α) · CE(student, labels)

Where:
    T     = temperature  (softens teacher distribution; recommended 2–4)
    α     = distillation weight  (recommended 0.5–0.9)
    KL    = mean KL divergence restricted to response token positions
    CE    = hard cross-entropy restricted to response token positions

Both teacher and student share the Qwen2.5-VL tokenizer so KL divergence is
computed over the same token space without any vocabulary projection.

Memory budget (RTX 4080 Super 16 GB):
    Teacher Qwen2.5-VL-7B (4-bit NF4) : ~6.0 GB
    Student AetherForge 128M (fp16)    : ~0.3 GB weights + ~0.6 GB optimizer
    Activations (batch=1, seq=256)     : ~1.0 GB
    Teacher logits buffer (fp16)       : ~80 MB per step
    Total                              : ~8–9 GB  (comfortable on 16 GB)

Usage:
    # Smoke-test (25 steps)
    conda run -n ml-torch python scripts/distill_aetherforge.py --test-run

    # Full run from built-in sample dataset
    conda run -n ml-torch python scripts/distill_aetherforge.py \\
        --config 128M --temperature 3.0 --alpha 0.7 --steps 5000

    # From your own JSONL  ({"instruction": "...", "response": "..."} per line)
    conda run -n ml-torch python scripts/distill_aetherforge.py \\
        --config 128M --data data/your_data.jsonl \\
        --temperature 3.0 --alpha 0.7 --steps 10000 --wandb
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge, MODEL_CONFIGS

TEACHER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Built-in sample dataset (240 instruction-response pairs)
# ---------------------------------------------------------------------------

BUILTIN_DATA = [
    {"instruction": "What is the Transformer architecture?",
     "response": "The Transformer uses self-attention to model dependencies between all "
                 "positions in a sequence simultaneously, replacing recurrent networks with "
                 "parallelisable multi-head attention layers."},
    {"instruction": "Explain sparse Mixture of Experts in one paragraph.",
     "response": "Sparse MoE routes each token to a small subset (top-k) of specialised "
                 "feed-forward networks called experts. A learned router assigns weights; "
                 "only the selected experts execute, so total compute per token stays low "
                 "while model capacity scales with the number of experts."},
    {"instruction": "What is knowledge distillation?",
     "response": "Knowledge distillation trains a smaller student model to mimic a larger "
                 "teacher. The student minimises KL divergence between its output distribution "
                 "and the teacher's soft probability distribution (temperature-scaled logits), "
                 "transferring the teacher's learned representations more efficiently than "
                 "training on hard labels alone."},
    {"instruction": "What is 4-bit NF4 quantisation?",
     "response": "NF4 maps model weights to a 4-bit grid whose levels are optimally spaced "
                 "for normally distributed weights. Double quantisation then quantises the "
                 "quantisation constants, saving ~0.4 bits/parameter with minimal accuracy loss."},
    {"instruction": "What is LoRA and how does it reduce VRAM?",
     "response": "LoRA inserts trainable low-rank matrices A and B into each attention "
                 "projection so only r×(d_in+d_out) parameters are updated instead of "
                 "d_in×d_out. This reduces trainable parameters by 10–100× and optimizer "
                 "state VRAM proportionally."},
    {"instruction": "Describe Flash Attention and its memory benefit.",
     "response": "Flash Attention rewrites the attention kernel to tile Q, K, V in SRAM, "
                 "avoiding materialising the full N×N attention matrix in HBM. Memory drops "
                 "from O(N²) to O(N) and wall-clock speed improves 2–4× for long sequences."},
    {"instruction": "What is Rotary Position Embedding (RoPE)?",
     "response": "RoPE encodes position by rotating pairs of hidden dimensions with "
                 "position-dependent angles, multiplying the query and key vectors "
                 "before the attention dot product. Unlike absolute embeddings, RoPE "
                 "preserves relative position information and generalises to sequences "
                 "longer than those seen during training."},
    {"instruction": "Explain the role of temperature in knowledge distillation.",
     "response": "Temperature T > 1 softens the teacher's output distribution before "
                 "computing KL divergence, revealing the teacher's confidence structure "
                 "across non-target classes. This richer signal helps the student learn "
                 "which classes are semantically related. The KL loss is scaled by T² "
                 "to maintain the gradient magnitude as T increases."},
    {"instruction": "Describe a hospital navigation task for a mobile robot.",
     "response": "The robot must localise itself in a dynamic environment with staff, "
                 "patients, and equipment. It plans collision-free paths to target rooms, "
                 "adjusts speed near pedestrians, and communicates via audio and lighting "
                 "while complying with hygiene and safety protocols."},
    {"instruction": "What is gradient checkpointing?",
     "response": "Gradient checkpointing saves memory by not storing all intermediate "
                 "activations during the forward pass. Instead, activations are recomputed "
                 "during the backward pass from saved checkpoints. This trades ~30% extra "
                 "compute for a large reduction in activation memory."},
] * 24   # 240 samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """
    Instruction-response dataset for distillation.

    Returns tokenized sequences with labels masked to -100 on prompt tokens
    so both KL and CE losses are restricted to response positions only.
    """

    def __init__(self, tokenizer, path: Optional[str], max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        if path and Path(path).exists():
            with open(path) as f:
                self.data = [json.loads(l) for l in f if l.strip()]
            print(f"Loaded {len(self.data)} samples from {path}")
        else:
            if path:
                print(f"Warning: {path} not found — using built-in dataset.")
            self.data = BUILTIN_DATA
            print(f"Using built-in dataset ({len(self.data)} samples)")

    def _prompt_len(self, instruction: str) -> int:
        """Tokenised length of the user turn (used to compute the mask offset)."""
        msgs  = [{"role": "user", "content": instruction}]
        tmpl  = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return len(self.tokenizer.encode(tmpl, add_special_tokens=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        msgs = [
            {"role": "user",      "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]
        full = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        ids    = enc["input_ids"].squeeze(0)
        labels = ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask prompt (instruction) tokens — loss only on response
        prompt_len = min(self._prompt_len(item["instruction"]), self.max_length)
        labels[:prompt_len] = -100

        return {
            "input_ids":      ids,
            "labels":         labels,
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def distill_loss(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    labels:         torch.Tensor,   # [B, T]  (-100 = ignore)
    temperature:    float,
    alpha:          float,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (total_loss, kl_loss_item, ce_loss_item).

    KL divergence is computed on positions where labels != -100 (response only).
    Temperature scaling follows Hinton et al. 2015: multiply KL by T².
    """
    B, T, V = student_logits.shape
    valid = (labels != -100)   # [B, T]

    if valid.any():
        V_s = student_logits.shape[-1]
        s_valid = student_logits[valid]           # [N, V_student]
        # Teacher vocab may include visual tokens (Qwen2.5-VL head = 152064).
        # Slice to student vocab size — visual token logits never appear in
        # text sequences so excluding them from KL is correct.
        t_valid = teacher_logits[valid][..., :V_s]  # [N, V_student]

        log_p_s  = F.log_softmax(s_valid / temperature, dim=-1)
        p_t      = F.softmax(t_valid / temperature, dim=-1)

        # KL(teacher || student) per token, summed over vocab, mean over N tokens
        kl_per   = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)  # [N]
        kl       = kl_per.mean() * (temperature ** 2)
    else:
        kl = torch.tensor(0.0, device=student_logits.device)

    # Hard cross-entropy on response tokens
    ce = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
    )

    total = alpha * kl + (1.0 - alpha) * ce
    return total, kl.item(), ce.item()


# ---------------------------------------------------------------------------
# Helpers (scheduler, checkpoint, loss curve) — same style as train_aetherforge
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, loss, out_dir: Path) -> Path:
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "loss": loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt / "state.pt")
    (ckpt / "config.json").write_text(json.dumps({"step": step, "loss": loss}))
    return ckpt


def save_loss_curve(log_path: Path, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(log_path)
        if df.empty:
            return

        BG, SURF = "#0D1117", "#161B22"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor(BG)
        for ax in axes:
            ax.set_facecolor(SURF)
            ax.tick_params(colors="#E6EDF3")
            ax.xaxis.label.set_color("#E6EDF3")
            ax.yaxis.label.set_color("#E6EDF3")
            ax.title.set_color("#E6EDF3")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363D")

        axes[0].plot(df["step"], df["kl_loss"], color="#58A6FF", lw=1.2, alpha=0.5, label="KL")
        axes[0].plot(df["step"], df["ce_loss"], color="#E3B341", lw=1.2, alpha=0.5, label="CE")
        axes[0].plot(df["step"], df["total_loss"], color="#E6EDF3", lw=2.0, label="total")
        if len(df) >= 5:
            ma = df["total_loss"].rolling(5, min_periods=1).mean()
            axes[0].plot(df["step"], ma, color="#3FB950", lw=2, linestyle="--", label="MA-5")
        axes[0].set(xlabel="Step", ylabel="Loss", title="Distillation Loss")
        axes[0].legend(framealpha=0.3)
        axes[0].grid(alpha=0.2, color="#30363D")

        axes[1].fill_between(df["step"], df["lr"], alpha=0.4, color="#8B949E")
        axes[1].plot(df["step"], df["lr"], color="#8B949E", lw=1.5)
        axes[1].set(xlabel="Step", ylabel="LR", title="Learning Rate")
        axes[1].grid(alpha=0.2, color="#30363D")

        axes[2].plot(df["step"], df["vram_gb"], color="#3FB950", lw=1.5)
        axes[2].axhline(16.0, color="#F85149", lw=1, linestyle="--", alpha=0.6, label="16 GB")
        axes[2].set(xlabel="Step", ylabel="VRAM (GB)", title="GPU Memory")
        axes[2].legend(framealpha=0.3)
        axes[2].grid(alpha=0.2, color="#30363D")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor=BG)
        plt.close()
        print(f"Loss curve saved to {out_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(device: str):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"Loading teacher {TEACHER_ID} in 4-bit NF4 ...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEACHER_ID, quantization_config=quant,
        device_map="auto", trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # len(tokenizer) includes special tokens added on top of tokenizer.vocab_size
    # (e.g. Qwen2.5: vocab_size=151643, len=151665, max_id=151664).
    # The student embedding table must cover the maximum token ID.
    print(f"Teacher tokenizer: vocab_size={tokenizer.vocab_size}  "
          f"len={len(tokenizer)}  (using len for embedding table)")
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"Teacher loaded — VRAM so far: {vram:.2f} GB")
    return teacher, tokenizer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def distill(
    data_path:          Optional[str] = None,
    model_config:       str   = "128M",
    student_checkpoint: Optional[str] = None,
    max_length:         int   = 256,
    batch_size:         int   = 1,
    grad_accum:         int   = 8,
    lr:                 float = 2e-4,
    weight_decay:       float = 0.1,
    steps:              int   = 5000,
    warmup_steps:       int   = 200,
    save_steps:         int   = 500,
    log_every:          int   = 10,
    temperature:        float = 3.0,
    alpha:              float = 0.7,
    output_dir:         str   = "outputs/aetherforge_distill",
    resume:             Optional[str] = None,
    test_run:           bool  = False,
    use_wandb:          bool  = False,
    wandb_project:      str   = "aetherforge",
):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_run:
        steps, warmup_steps, save_steps, batch_size = 25, 5, 15, 1

    # ── Teacher ──────────────────────────────────────────────────────────
    teacher, tokenizer = load_teacher(device)
    vocab_size = len(tokenizer)   # covers all special tokens; max_id < len(tok)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = DistillDataset(tokenizer, data_path, max_length)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # ── Student ──────────────────────────────────────────────────────────
    cfg = dict(MODEL_CONFIGS[model_config])
    cfg["vocab_size"] = vocab_size    # match teacher's token space
    student = AetherForge(**cfg).to(device)

    if student_checkpoint and Path(student_checkpoint).exists():
        ckpt = torch.load(student_checkpoint, map_location=device, weights_only=True)
        # Accept either raw state_dict or checkpoint dict with "model" key
        sd = ckpt.get("model", ckpt)
        try:
            student.load_state_dict(sd, strict=False)
            print(f"Loaded student weights from {student_checkpoint}")
        except RuntimeError as e:
            print(f"Warning: student checkpoint load failed ({e}); using random init")

    if resume:
        state = torch.load(Path(resume) / "state.pt", map_location=device,
                           weights_only=True)
        student.load_state_dict(state["model"])
        print(f"Resumed from {resume}  (step {state['step']})")

    print(f"\nStudent [{model_config}]  {student.param_count()}  |  "
          f"vocab {vocab_size}  |  α={alpha}  T={temperature}")

    vram_after = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"Both models loaded — VRAM: {vram_after:.2f} GB")

    # ── Optimiser + schedule ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, warmup_steps, steps)
    scaler    = GradScaler("cuda") if device == "cuda" else None

    start_step = state["step"] if resume else 0
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    # ── W&B ──────────────────────────────────────────────────────────────
    if use_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, config={
                "model_config": model_config, "vocab_size": vocab_size,
                "temperature": temperature, "alpha": alpha,
                "lr": lr, "steps": steps, "max_length": max_length,
            })
        except ImportError:
            print("wandb not installed — skipping.")
            use_wandb = False

    # ── CSV log ──────────────────────────────────────────────────────────
    log_path   = out_dir / "distill_log.csv"
    log_file   = open(log_path, "a" if resume else "w", newline="")
    log_writer = csv.writer(log_file)
    if not resume:
        log_writer.writerow(["step", "total_loss", "kl_loss", "ce_loss",
                              "lr", "vram_gb", "tokens_per_sec", "elapsed_sec"])

    # ── Training loop ─────────────────────────────────────────────────────
    student.train()
    global_step   = start_step
    accum_total   = 0.0
    accum_kl      = 0.0
    accum_ce      = 0.0
    accum_tokens  = 0
    optimizer.zero_grad()
    t_start = time.time()
    t_log   = time.time()
    data_iter = iter(loader)

    while global_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ── Teacher forward (no grad, fp16) ──────────────────────────────
        with torch.no_grad():
            teacher_out    = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_out.logits.float()   # [B, T, V] in fp32

        # ── Student forward + loss ────────────────────────────────────────
        if device == "cuda":
            with autocast("cuda"):
                student_logits = student(input_ids)        # [B, T, V]
                total, kl, ce  = distill_loss(
                    student_logits.float(), teacher_logits,
                    labels, temperature, alpha,
                )
                scaled = total / grad_accum
            scaler.scale(scaled).backward()
        else:
            student_logits = student(input_ids)
            total, kl, ce  = distill_loss(
                student_logits.float(), teacher_logits,
                labels, temperature, alpha,
            )
            (total / grad_accum).backward()

        accum_total  += total.item()
        accum_kl     += kl
        accum_ce     += ce
        accum_tokens += (labels != -100).sum().item()
        global_step  += 1

        if global_step % grad_accum == 0:
            if device == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if device == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if global_step % log_every == 0:
            now          = time.time()
            elapsed      = now - t_start
            interval     = now - t_log
            tok_per_sec  = accum_tokens / max(interval, 1e-6)
            remaining    = (steps - global_step) * (interval / log_every)
            eta          = f"{int(remaining//60)}m{int(remaining%60)}s"
            lr_now       = scheduler.get_last_lr()[0]
            vram         = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
            n            = log_every

            avg_total = accum_total / n
            avg_kl    = accum_kl    / n
            avg_ce    = accum_ce    / n

            print(
                f"step {global_step:5d}/{steps} | "
                f"loss {avg_total:.4f} "
                f"(KL {avg_kl:.4f}  CE {avg_ce:.4f}) | "
                f"lr {lr_now:.2e} | "
                f"vram {vram:.2f}GB | "
                f"{tok_per_sec:.0f} tok/s | "
                f"eta {eta}"
            )

            log_writer.writerow([
                global_step,
                round(avg_total, 6), round(avg_kl, 6), round(avg_ce, 6),
                round(lr_now, 8), round(vram, 3),
                round(tok_per_sec, 1), round(elapsed, 1),
            ])
            log_file.flush()

            if use_wandb:
                import wandb
                wandb.log({"total_loss": avg_total, "kl_loss": avg_kl,
                           "ce_loss": avg_ce, "lr": lr_now,
                           "vram_gb": vram, "tok_per_sec": tok_per_sec},
                          step=global_step)

            accum_total = accum_kl = accum_ce = accum_tokens = 0.0
            t_log       = now

        if global_step % save_steps == 0:
            ckpt = save_checkpoint(student, optimizer, scheduler,
                                   global_step, avg_total, out_dir)
            print(f"  saved → {ckpt}")

    log_file.close()
    if use_wandb:
        import wandb
        wandb.finish()

    final = out_dir / "final"
    final.mkdir(exist_ok=True)
    torch.save(student.state_dict(), final / "model.pt")
    cfg_out = dict(MODEL_CONFIGS[model_config])
    cfg_out["vocab_size"] = vocab_size   # len(tokenizer), not tokenizer.vocab_size
    (final / "config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"\nDistillation complete.  Model: {final}/model.pt")
    print(f"Config: {final}/config.json")
    print(f"Log:    {log_path}")

    save_loss_curve(log_path, out_dir / "distill_loss_curve.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Knowledge distillation: Qwen2.5-VL-7B → AetherForge",
    )
    p.add_argument("--data",          default=None,
                   help="JSONL dataset (instruction/response). Uses built-in 240 "
                        "samples if omitted.")
    p.add_argument("--config",        default="128M", choices=list(MODEL_CONFIGS),
                   help="AetherForge student size.")
    p.add_argument("--temperature",   type=float, default=3.0,
                   help="Distillation temperature T (>1 softens teacher; try 2–4).")
    p.add_argument("--alpha",         type=float, default=0.7,
                   help="KL loss weight α. CE weight = 1-α. Try 0.5–0.9.")
    p.add_argument("--max-length",    type=int,   default=256,
                   help="Token sequence length.")
    p.add_argument("--batch-size",    type=int,   default=1)
    p.add_argument("--grad-accum",    type=int,   default=8,
                   help="Gradient accumulation (eff. batch = batch × accum).")
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--weight-decay",  type=float, default=0.1)
    p.add_argument("--steps",         type=int,   default=5000)
    p.add_argument("--warmup-steps",  type=int,   default=200)
    p.add_argument("--save-steps",    type=int,   default=500)
    p.add_argument("--log-every",     type=int,   default=10)
    p.add_argument("--output-dir",    default="outputs/aetherforge_distill")
    p.add_argument("--resume",        default=None,
                   help="Checkpoint directory to resume from.")
    p.add_argument("--wandb",         dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", default="aetherforge")
    p.add_argument("--student-checkpoint", default=None, dest="student_checkpoint",
                   help="Path to a pretrained AetherForge .pt file to warm-start "
                        "the student. Must match --config vocab/dimensions.")
    p.add_argument("--test-run",      action="store_true",
                   help="25 steps — fast pipeline verification.")
    args = p.parse_args()

    distill(
        data_path          = args.data,
        model_config       = args.config,
        student_checkpoint = args.student_checkpoint,
        max_length     = args.max_length,
        batch_size     = args.batch_size,
        grad_accum     = args.grad_accum,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        steps          = args.steps,
        warmup_steps   = args.warmup_steps,
        save_steps     = args.save_steps,
        log_every      = args.log_every,
        temperature    = args.temperature,
        alpha          = args.alpha,
        output_dir     = args.output_dir,
        resume         = args.resume,
        test_run       = args.test_run,
        use_wandb      = args.use_wandb,
        wandb_project  = args.wandb_project,
    )


if __name__ == "__main__":
    main()
