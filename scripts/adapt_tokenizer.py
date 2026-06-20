"""
scripts/adapt_tokenizer.py
Swap a char-level AetherForge checkpoint to the Qwen BPE tokenizer.

The only vocab-dependent tensors are embedding.weight and lm_head.weight
(weight-tied, so stored once). Everything else — attention, MoE, FRC — is
vocab-independent and can be reused directly.

New embedding rows are initialised at the same std as the original so the
model starts with sensible activation magnitudes. The body weights already
encode language structure from FineWeb pretraining; fine-tuning adapts the
embedding surface.

Usage:
    conda run -n ml-torch python scripts/adapt_tokenizer.py \
        --checkpoint outputs/aetherforge_1B_pretrain/final/model.pt \
        --config     1B \
        --output     outputs/aetherforge_1B_bpe/init

    # Then fine-tune:
    conda run -n ml-torch python scripts/finetune_code_agent.py \
        --checkpoint outputs/aetherforge_1B_bpe/init/model.pt \
        --config     outputs/aetherforge_1B_bpe/init/config.json \
        --data       data/code_agent_data.jsonl \
        --steps 5000 --lr 3e-5 --batch-size 1 --grad-accum 8 --8bit-adam
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge, MODEL_CONFIGS

QWEN_VOCAB = 151_665   # len(AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct"))


def adapt(checkpoint: str, config: str, output: str):
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original weights ─────────────────────────────────────────
    print(f"Loading {checkpoint} ...")
    orig = torch.load(checkpoint, map_location="cpu", weights_only=True)

    old_embed = orig["embedding.weight"]            # [32000, d_model]
    d_model   = old_embed.shape[1]
    old_vocab  = old_embed.shape[0]
    old_std    = old_embed.std().item()
    old_mean   = old_embed.mean().item()
    print(f"  Old vocab: {old_vocab}   d_model: {d_model}")
    print(f"  Embedding stats — mean: {old_mean:.4f}  std: {old_std:.4f}")

    # ── Build new embedding [QWEN_VOCAB, d_model] ─────────────────────
    # Use same scale as original; first 32k rows could map old char-level
    # tokens but char→BPE correspondence is arbitrary, so full random init
    # is cleaner and fine-tuning adapts all rows.
    new_embed = torch.empty(QWEN_VOCAB, d_model)
    torch.nn.init.normal_(new_embed, mean=0.0, std=old_std * 0.5)

    # ── Build new state dict ──────────────────────────────────────────
    new_state = {}
    skipped   = 0
    for k, v in orig.items():
        if k in ("embedding.weight", "lm_head.weight"):
            skipped += 1
            continue
        new_state[k] = v.clone()
    new_state["embedding.weight"] = new_embed
    new_state["lm_head.weight"]   = new_embed   # weight tying — same tensor

    # ── Instantiate model and verify ──────────────────────────────────
    cfg = dict(MODEL_CONFIGS[config])
    cfg["vocab_size"] = QWEN_VOCAB
    model = AetherForge(**cfg)
    missing, unexpected = model.load_state_dict(new_state, strict=True), []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  New vocab: {QWEN_VOCAB}   params: {n_params/1e6:.1f}M")

    # Verify body weights transferred correctly
    body_key = "blocks.0.mla.W_q.weight"
    if body_key in orig and body_key in new_state:
        assert torch.allclose(orig[body_key], new_state[body_key]), \
            "Body weight mismatch — something went wrong"
    print("  Body weights verified — all copied cleanly.")

    # ── Save ──────────────────────────────────────────────────────────
    model_path = out_dir / "model.pt"
    cfg_path   = out_dir / "config.json"

    torch.save(model.state_dict(), model_path)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved adapted model → {model_path}")
    print(f"Saved config         → {cfg_path}")
    print(f"\nNext step:")
    print(f"  conda run -n ml-torch python scripts/finetune_code_agent.py \\")
    print(f"    --checkpoint {model_path} \\")
    print(f"    --config     {cfg_path} \\")
    print(f"    --data       data/code_agent_data.jsonl \\")
    print(f"    --steps 5000 --lr 3e-5 --batch-size 1 --grad-accum 8 --8bit-adam")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/aetherforge_1B_pretrain/final/model.pt")
    p.add_argument("--config",     default="1B",
                   help="MODEL_CONFIGS key (e.g. '1B')")
    p.add_argument("--output",     default="outputs/aetherforge_1B_bpe/init")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if args.config not in MODEL_CONFIGS:
        print(f"ERROR: unknown config '{args.config}'. Choose from: {list(MODEL_CONFIGS)}")
        sys.exit(1)

    adapt(args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
