"""
AetherForge pretraining — trains the custom 128M–13B model from scratch.

Data sources (in priority order):
  1. HuggingFace streaming dataset  (--stream; e.g. FineWeb, DCLM — no download)
  2. Local JSONL file               (--data path/to/file.jsonl)
  3. Synthetic random tokens        (--test-run; no data needed)

Scaling:
  Single GPU:   python  scripts/train_aetherforge.py  [args]
  Multi-GPU:    torchrun --nproc_per_node=N  scripts/train_aetherforge.py  [args]

Usage examples:
    # Smoke-test — no data, no GPU required
    conda run -n ml-torch python scripts/train_aetherforge.py --test-run

    # FineWeb streaming  (no disk space needed — samples on the fly)
    conda run -n ml-torch python scripts/train_aetherforge.py \\
        --stream --dataset HuggingFaceFW/fineweb \\
        --dataset-config sample-10BT --steps 10000

    # Local JSONL + gradient checkpointing
    conda run -n ml-torch python scripts/train_aetherforge.py \\
        --data data/synthetic_data.jsonl --config 1B \\
        --gradient-checkpointing --steps 5000

    # Multi-GPU DDP  (4 × RTX 4080 Super)
    torchrun --nproc_per_node=4 scripts/train_aetherforge.py \\
        --stream --config 7B --batch-size 2 --steps 50000 --wandb
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, str]:
    """Returns (local_rank, world_size, device). Call before any CUDA ops."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, device
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), f"cuda:{local_rank}"


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def raw_model(model) -> AetherForge:
    """Unwrap DDP to reach the underlying AetherForge module."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticRandomDataset(Dataset):
    """Random token IDs — instant sanity check, no files needed."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 2000):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_samples  = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, _):
        ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return ids[:-1], ids[1:]


class JSONLTextDataset(Dataset):
    """
    Instruction/response pairs from a JSONL file.
    Supports {"instruction"/"response"} and {"text"} formats.

    Tokenizer priority:
      1. HuggingFace tokenizer (--tokenizer HF-id or local path)
      2. Character-level fallback (ord(c) % vocab_size) — no deps
    """
    def __init__(self, path: str, seq_len: int, vocab_size: int, tokenizer=None):
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.tokenizer  = tokenizer
        self.samples: list[str] = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "instruction" in obj and "response" in obj:
                    self.samples.append(obj["instruction"] + " " + obj["response"])
                elif "text" in obj:
                    self.samples.append(obj["text"])

        tok_name = type(tokenizer).__name__ if tokenizer else "char-level"
        print(f"Loaded {len(self.samples)} samples from {path}  [{tok_name}]")

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            return ids[: self.seq_len + 1]
        return [ord(c) % self.vocab_size for c in text[: self.seq_len + 1]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids     = self._encode(self.samples[idx])
        pad_len = self.seq_len + 1 - len(ids)
        ids     = ids + [0] * pad_len
        ids     = torch.tensor(ids[:self.seq_len + 1], dtype=torch.long)
        return ids[:-1], ids[1:]


class StreamingHFDataset(IterableDataset):
    """
    Streams text from any HuggingFace dataset without downloading it.

    Packs tokenized text into fixed-length chunks of seq_len tokens.
    Automatically reshards across DDP workers so each GPU sees different data.

    Recommended datasets:
      --dataset HuggingFaceFW/fineweb  --dataset-config sample-10BT   (English web)
      --dataset tiiuae/falcon-refinedweb                               (multilingual)
      --dataset allenai/dolma           --dataset-config v1_6-sample   (diverse)

    Falls back to char-level encoding if no tokenizer is provided.
    """
    def __init__(
        self,
        dataset_name:   str,
        dataset_config: str | None,
        dataset_split:  str,
        seq_len:        int,
        vocab_size:     int,
        tokenizer=None,
        rank:           int = 0,
        world_size:     int = 1,
        max_samples:    int | None = None,
    ):
        self.seq_len      = seq_len
        self.vocab_size   = vocab_size
        self.tokenizer    = tokenizer
        self.rank         = rank
        self.world_size   = world_size
        self.max_samples  = max_samples

        from datasets import load_dataset
        self.ds = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            streaming=True,
            trust_remote_code=True,
        )
        print(f"Streaming dataset: {dataset_name}"
              f"{f'/{dataset_config}' if dataset_config else ''}"
              f"  split={dataset_split}")

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) % self.vocab_size for c in text]

    def __iter__(self):
        buf: list[int] = []
        seen = 0
        for i, item in enumerate(self.ds):
            # Shard across DDP ranks
            if self.world_size > 1 and i % self.world_size != self.rank:
                continue

            text = item.get("text", "") or item.get("content", "")
            if not text:
                continue
            buf.extend(self._encode(text))

            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len:]
                ids   = torch.tensor(chunk, dtype=torch.long)
                yield ids[:-1], ids[1:]
                seen += 1
                if self.max_samples and seen >= self.max_samples:
                    return


# ---------------------------------------------------------------------------
# LR schedule, checkpointing, loss curve
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler,
                    step: int, loss: float, out_dir: Path) -> Path:
    ckpt_dir = out_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rm = raw_model(model)
    torch.save({
        "step":            step,
        "loss":            loss,
        "model_state":     rm.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, ckpt_dir / "state.pt")
    cfg = {
        "vocab_size": rm.embedding.num_embeddings,
        "d_model":    rm.embedding.embedding_dim,
        "n_layers":   len(rm.blocks),
        "n_heads":    rm.blocks[0].attn.n_heads,
        "latent_dim": rm.blocks[0].attn.kv_compress.out_features,
        "n_experts":  rm.blocks[0].moe.n_experts,
        "top_k":      rm.blocks[0].moe.top_k,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt_dir


def load_checkpoint(path: str, model, optimizer, scheduler) -> int:
    state = torch.load(Path(path) / "state.pt", weights_only=False)
    raw_model(model).load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    print(f"Resumed from step {state['step']}  (loss {state['loss']:.4f})")
    return state["step"]


def save_loss_curve(log_path: Path, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df  = pd.read_csv(log_path)
        if df.empty:
            return
        BG = "#0D1117"; SURF = "#161B22"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor(BG)
        for ax in axes:
            ax.set_facecolor(SURF)
            ax.tick_params(colors="#E6EDF3")
            ax.xaxis.label.set_color("#E6EDF3"); ax.yaxis.label.set_color("#E6EDF3")
            ax.title.set_color("#E6EDF3")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")

        axes[0].plot(df["step"], df["lm_loss"], color="#58A6FF", lw=1, alpha=0.5, label="LM")
        axes[0].plot(df["step"], df["balance_loss"], color="#E3B341", lw=1, alpha=0.5, label="balance")
        axes[0].plot(df["step"], df["loss"], color="#E6EDF3", lw=2, label="total")
        if len(df) >= 10:
            ma = df["loss"].rolling(10, min_periods=1).mean()
            axes[0].plot(df["step"], ma, color="#3FB950", lw=2, linestyle="--", label="MA-10")
        axes[0].set(xlabel="Step", ylabel="Loss", title="Training Loss")
        axes[0].legend(framealpha=0.3); axes[0].grid(alpha=0.2, color="#30363D")

        axes[1].fill_between(df["step"], df["lr"], alpha=0.4, color="#8B949E")
        axes[1].plot(df["step"], df["lr"], color="#8B949E", lw=1.5)
        axes[1].set(xlabel="Step", ylabel="LR", title="Learning Rate")
        axes[1].grid(alpha=0.2, color="#30363D")

        axes[2].plot(df["step"], df["vram_gb"], color="#3FB950", lw=1.5)
        axes[2].axhline(16.0, color="#F85149", lw=1, linestyle="--", alpha=0.6, label="16 GB")
        axes[2].set(xlabel="Step", ylabel="VRAM (GB)", title="GPU Memory")
        axes[2].legend(framealpha=0.3); axes[2].grid(alpha=0.2, color="#30363D")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor=BG)
        plt.close()
        print(f"Loss curve saved to {out_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    # Data
    data_path:              str | None = None,
    stream:                 bool  = False,
    dataset_name:           str   = "HuggingFaceFW/fineweb",
    dataset_config:         str | None = "sample-10BT",
    dataset_split:          str   = "train",
    # Model
    model_config:           str | None = None,
    vocab_size:             int   = 32000,
    d_model:                int   = 512,
    n_layers:               int   = 6,
    n_heads:                int   = 8,
    latent_dim:             int   = 128,
    n_experts:              int   = 8,
    top_k:                  int   = 2,
    gradient_checkpointing: bool  = False,
    # Training
    seq_len:                int   = 256,
    batch_size:             int   = 4,
    grad_accum:             int   = 4,
    lr:                     float = 3e-4,
    weight_decay:           float = 0.1,
    balance_alpha:          float = 0.01,
    steps:                  int   = 1000,
    warmup_steps:           int   = 100,
    save_steps:             int   = 500,
    log_every:              int   = 10,
    # Output
    output_dir:             str   = "outputs/aetherforge_pretrain",
    resume:                 str | None = None,
    tokenizer_id:           str | None = None,
    test_run:               bool  = False,
    use_wandb:              bool  = False,
    wandb_project:          str   = "aetherforge",
    use_8bit_adam:          bool  = False,
):
    # ── DDP init ──────────────────────────────────────────────────────────
    local_rank, world_size, device = setup_ddp()
    main = is_main(local_rank)

    out_dir = Path(output_dir)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)

    if test_run:
        steps, warmup_steps, save_steps = 50, 5, 25
        batch_size, seq_len = 2, 64
        stream = False

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = None
    if tokenizer_id:
        try:
            from transformers import AutoTokenizer
            tokenizer  = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            vocab_size = tokenizer.vocab_size
            if main:
                print(f"Tokenizer: {tokenizer_id}  (vocab {vocab_size})")
        except Exception as e:
            if main:
                print(f"Tokenizer load failed ({e}) — char-level fallback.")

    # ── Dataset ───────────────────────────────────────────────────────────
    if stream:
        dataset = StreamingHFDataset(
            dataset_name, dataset_config, dataset_split,
            seq_len, vocab_size, tokenizer,
            rank=local_rank, world_size=world_size,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size,
            num_workers=0, pin_memory=(device != "cpu"),
        )
    elif data_path and Path(data_path).exists():
        dataset = JSONLTextDataset(data_path, seq_len, vocab_size, tokenizer)
        sampler = DistributedSampler(dataset, world_size, local_rank) if world_size > 1 else None
        loader  = DataLoader(
            dataset, batch_size=batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=min(2, os.cpu_count() or 1), pin_memory=(device != "cpu"),
        )
    else:
        if data_path and main:
            print(f"Warning: {data_path} not found — using random token data.")
        n_samples = max(steps * batch_size, 2000)
        dataset   = SyntheticRandomDataset(vocab_size, seq_len, n_samples)
        sampler   = DistributedSampler(dataset, world_size, local_rank) if world_size > 1 else None
        loader    = DataLoader(
            dataset, batch_size=batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=0, pin_memory=(device != "cpu"),
        )

    # ── Model ─────────────────────────────────────────────────────────────
    if model_config and model_config in MODEL_CONFIGS:
        cfg = dict(MODEL_CONFIGS[model_config])
        cfg["vocab_size"] = vocab_size
        model = AetherForge(**cfg).to(device)
    else:
        model = AetherForge(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, latent_dim=latent_dim, n_experts=n_experts, top_k=top_k,
        ).to(device)

    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if main:
            print("Gradient checkpointing: ON")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if main:
        rm = raw_model(model)
        print(f"\nAetherForge [{model_config or 'custom'}] — {rm.param_count()} "
              f"on {device}  (world_size={world_size})")
        if gradient_checkpointing:
            print("Gradient checkpointing: ON  (lower VRAM at ~30% compute cost)")

    # ── Optimiser & schedule ──────────────────────────────────────────────
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                raw_model(model).parameters(), lr=lr,
                weight_decay=weight_decay, betas=(0.9, 0.95),
            )
            if main:
                print("Optimizer: AdamW8bit (bitsandbytes) — ~8× less VRAM for states")
        except ImportError:
            print("Warning: bitsandbytes not found, falling back to fp32 AdamW.")
            optimizer = torch.optim.AdamW(
                raw_model(model).parameters(), lr=lr,
                weight_decay=weight_decay, betas=(0.9, 0.95),
            )
    else:
        optimizer = torch.optim.AdamW(
            raw_model(model).parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
        )
    scheduler = build_scheduler(optimizer, warmup_steps, steps)
    scaler    = GradScaler("cuda") if "cuda" in device else None
    criterion = nn.CrossEntropyLoss()

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer, scheduler)

    # ── W&B (main rank only) ──────────────────────────────────────────────
    if use_wandb and main:
        try:
            import wandb
            wandb.init(project=wandb_project, config={
                "model_config": model_config, "world_size": world_size,
                "gradient_checkpointing": gradient_checkpointing,
                "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                "n_experts": n_experts, "lr": lr, "seq_len": seq_len,
                "balance_alpha": balance_alpha,
                "tokenizer": tokenizer_id or "char-level",
                "stream": stream, "dataset": dataset_name if stream else data_path,
            })
        except ImportError:
            print("wandb not installed — skipping.")
            use_wandb = False

    # ── CSV log (main rank only) ──────────────────────────────────────────
    log_path   = out_dir / "training_log.csv"
    log_file   = open(log_path, "a" if resume else "w", newline="") if main else None
    log_writer = csv.writer(log_file) if main else None
    if main and not resume:
        log_writer.writerow(["step", "loss", "lm_loss", "balance_loss",
                              "lr", "vram_gb", "tokens_per_sec", "elapsed_sec"])

    # ── Loop ──────────────────────────────────────────────────────────────
    model.train()
    global_step    = start_step
    accum_loss     = accum_lm_loss = accum_bal_loss = 0.0
    accum_tokens   = 0
    disp_loss      = 0.0
    optimizer.zero_grad()
    t_start = t_log = time.time()
    data_iter = iter(loader)

    while global_step < steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(global_step)
            data_iter = iter(loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        rm        = raw_model(model)

        if "cuda" in device:
            with autocast("cuda"):
                logits  = model(input_ids)
                lm_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                bal     = rm.load_balance_loss().to(device)
                loss    = (lm_loss + balance_alpha * bal) / grad_accum
            scaler.scale(loss).backward()
        else:
            logits  = model(input_ids)
            lm_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            bal     = rm.load_balance_loss()
            loss    = (lm_loss + balance_alpha * bal) / grad_accum
            loss.backward()

        accum_loss     += loss.item()
        accum_lm_loss  += lm_loss.item() / grad_accum
        accum_bal_loss += bal.item() / grad_accum
        accum_tokens   += labels.numel()
        global_step    += 1

        if global_step % grad_accum == 0:
            if "cuda" in device:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            if "cuda" in device:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if global_step % log_every == 0 and main:
            now          = time.time()
            elapsed      = now - t_start
            interval     = now - t_log
            tok_per_sec  = accum_tokens / max(interval, 1e-6)
            remaining    = (steps - global_step) * (interval / log_every)
            eta          = f"{int(remaining//60)}m{int(remaining%60)}s"
            lr_now       = scheduler.get_last_lr()[0]
            vram         = (torch.cuda.memory_allocated() / 1e9
                            if "cuda" in device else 0.0)
            # Mean per micro-step: accum already divided by grad_accum, so ×÷ cancel.
            disp_loss    = accum_loss     * grad_accum / log_every
            disp_lm      = accum_lm_loss  * grad_accum / log_every
            disp_bal     = accum_bal_loss * grad_accum / log_every

            print(
                f"step {global_step:5d}/{steps} | "
                f"loss {disp_loss:.4f} (lm {disp_lm:.4f} bal {disp_bal:.4f}) | "
                f"lr {lr_now:.2e} | "
                f"vram {vram:.2f}GB | "
                f"{tok_per_sec:.0f} tok/s | "
                f"eta {eta}"
                + (f" | GPUs×{world_size}" if world_size > 1 else "")
            )
            log_writer.writerow([global_step, round(disp_loss, 6),
                                  round(disp_lm, 6), round(disp_bal, 6),
                                  round(lr_now, 8), round(vram, 3),
                                  round(tok_per_sec, 1), round(elapsed, 1)])
            log_file.flush()

            if use_wandb:
                import wandb
                wandb.log({"loss": disp_loss, "lm_loss": disp_lm,
                           "balance_loss": disp_bal, "lr": lr_now,
                           "vram_gb": vram, "tok_per_sec": tok_per_sec},
                          step=global_step)

            accum_loss = accum_lm_loss = accum_bal_loss = 0.0
            accum_tokens = 0
            t_log = now

        if global_step % save_steps == 0 and main:
            ckpt = save_checkpoint(model, optimizer, scheduler,
                                   global_step, disp_loss, out_dir)
            print(f"  saved → {ckpt}")

    # ── Finalise ──────────────────────────────────────────────────────────
    if main:
        log_file.close()
        final_dir = out_dir / "final"
        final_dir.mkdir(exist_ok=True)
        torch.save(raw_model(model).state_dict(), final_dir / "model.pt")
        print(f"\nTraining complete.  Model: {final_dir}/model.pt")
        print(f"Log: {log_path}")
        if use_wandb:
            import wandb
            wandb.finish()
        save_loss_curve(log_path, out_dir / "loss_curve.png")

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="AetherForge pretraining  (single-GPU or torchrun multi-GPU)",
    )
    # Data
    p.add_argument("--data", default=None,
                   help="JSONL file (instruction/response or text). "
                        "Omit to use random smoke-test data.")
    p.add_argument("--stream", action="store_true",
                   help="Stream data from a HuggingFace dataset (no download).")
    p.add_argument("--dataset",        default="HuggingFaceFW/fineweb",
                   help="HF dataset repo id  (only used with --stream).")
    p.add_argument("--dataset-config", default="sample-10BT",
                   help="HF dataset config/subset  (only used with --stream).")
    p.add_argument("--dataset-split",  default="train",
                   help="HF dataset split  (only used with --stream).")
    p.add_argument("--tokenizer", dest="tokenizer_id", default=None,
                   help="HF tokenizer id or local path.  "
                        "Omit for char-level fallback.")
    # Model — preset or manual
    p.add_argument("--config", dest="model_config", default=None,
                   choices=list(MODEL_CONFIGS),
                   help=f"Model size preset. Choices: {list(MODEL_CONFIGS)}")
    p.add_argument("--vocab-size",  type=int, default=32000)
    p.add_argument("--d-model",     type=int, default=512)
    p.add_argument("--n-layers",    type=int, default=6)
    p.add_argument("--n-heads",     type=int, default=8)
    p.add_argument("--latent-dim",  type=int, default=128)
    p.add_argument("--n-experts",   type=int, default=8)
    p.add_argument("--top-k",       type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Recompute activations during backward  "
                        "(saves ~40%% VRAM at ~30%% compute cost). "
                        "Essential for 1B+ on a single GPU.")
    # Training
    p.add_argument("--seq-len",      type=int,   default=256)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=4,
                   help="Gradient accumulation  (eff. batch = batch × accum × GPUs)")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--balance-alpha",type=float, default=0.01,
                   help="AQ-MoE load-balance loss weight  (0 to disable)")
    p.add_argument("--steps",        type=int,   default=1000)
    p.add_argument("--warmup-steps", type=int,   default=100)
    p.add_argument("--save-steps",   type=int,   default=500)
    p.add_argument("--log-every",    type=int,   default=10)
    # Output
    p.add_argument("--output-dir", default="outputs/aetherforge_pretrain")
    p.add_argument("--resume",     default=None,
                   help="Checkpoint directory to resume from.")
    # Flags
    p.add_argument("--test-run", action="store_true",
                   help="50 steps, tiny config — verify pipeline in ~30 s.")
    p.add_argument("--wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", default="aetherforge")
    p.add_argument("--8bit-adam", dest="use_8bit_adam", action="store_true",
                   help="Use bitsandbytes AdamW8bit — cuts optimizer VRAM by ~8×. "
                        "Essential for 1B+ models on 16 GB GPUs.")
    args = p.parse_args()

    train(
        data_path              = args.data,
        stream                 = args.stream,
        dataset_name           = args.dataset,
        dataset_config         = args.dataset_config,
        dataset_split          = args.dataset_split,
        model_config           = args.model_config,
        vocab_size             = args.vocab_size,
        d_model                = args.d_model,
        n_layers               = args.n_layers,
        n_heads                = args.n_heads,
        latent_dim             = args.latent_dim,
        n_experts              = args.n_experts,
        top_k                  = args.top_k,
        gradient_checkpointing = args.gradient_checkpointing,
        seq_len                = args.seq_len,
        batch_size             = args.batch_size,
        grad_accum             = args.grad_accum,
        lr                     = args.lr,
        weight_decay           = args.weight_decay,
        balance_alpha          = args.balance_alpha,
        steps                  = args.steps,
        warmup_steps           = args.warmup_steps,
        save_steps             = args.save_steps,
        log_every              = args.log_every,
        output_dir             = args.output_dir,
        resume                 = args.resume,
        tokenizer_id           = args.tokenizer_id,
        test_run               = args.test_run,
        use_wandb              = args.use_wandb,
        wandb_project          = args.wandb_project,
        use_8bit_adam          = args.use_8bit_adam,
    )


if __name__ == "__main__":
    main()
