# Changelog

All notable changes to AetherForge-AI are documented here.

## [0.1.0] — 2026-06-18

First tagged release. Covers the full Phase 1–3 build-out.

### Phase 1 — Foundation

- `aetherforge/model.py`: decoder-only transformer with MLAPlus (KV compression),
  SparseMoE (top-K routing), and ForgeReasoningCore (gated iterative refinement)
- `scripts/train_aetherforge.py`: pretraining loop with AMP, cosine LR, W&B, CSV log
- `scripts/finetune_qwen25_vl.py`: LoRA fine-tuning of Qwen2.5-VL-7B in 4-bit NF4
  (text and multimodal modes)
- `scripts/inference_qwen25_vl.py`: Qwen2.5-VL inference (text + vision, interactive)
- `scripts/evaluate_model.py`: text + vision benchmarks, perplexity, base vs LoRA diff
- `Training_Dashboard.ipynb`: 13-panel Palantir-grade monitoring dashboard
- `multimodal_example/`: 20 synthetic hospital-scene images + JSONL generator

### Phase 2 — Architecture Upgrades

- **RoPE**: replaced learned absolute positions with rotary embeddings; lazy freq
  cache supports 1M+ token context without re-training
- **Flash Attention 2**: `F.scaled_dot_product_attention(is_causal=True)` dispatches
  to FA2 CUDA kernels; replaces manual QK^T materialisation
- **AQ-MoE coupling matrix J**: learned E×E matrix captures expert-to-expert
  correlations; coupled router `s' = s + η(s @ J)`, init zeros for stable training;
  load-balance loss `L = E Σ(f_i · P_i)` added to training objective
- **FRC v2 memory buffer**: 64 learned KV slots; per-step gated soft-attention read;
  gate init zeros ensures memory starts silent and is learned in
- **FRC v2 tool hooks**: `register_tool(step, fn)` wires any Python callable into the
  reasoning chain mid-forward (retrieval, calculator, structured extraction, etc.)
- **MODEL_CONFIGS** presets: 128M / 1B / 7B / 13B; `AetherForge.from_config(name)`
- **Fine-tuning quality fixes**: response-only loss masking (prompt tokens → -100),
  linear warmup + cosine decay scheduler, `image_grid_thw` fix for Qwen2.5-VL
  multimodal encoding, loss display corrected
- `scripts/serve.py`: FastAPI inference server — `/generate`, `/stream` (SSE),
  `/chat`; serves AetherForge or Qwen2.5-VL + LoRA backend
- `docs/architecture.png`: programmatic dark-theme architecture diagram
- `docs/manuscript.md`: full technical paper with honest Phase 2/3 roadmap

### Phase 3 — Scale

- **Gradient checkpointing**: `model.enable_gradient_checkpointing()` wraps each
  AetherForge block in `torch.utils.checkpoint`; `--gradient-checkpointing` flag
  in train script; ~40% VRAM reduction at ~30% extra compute
- **Multi-GPU DDP**: `torchrun --nproc_per_node=N scripts/train_aetherforge.py ...`
  — detects `LOCAL_RANK`, inits nccl process group, wraps model in DDP,
  `DistributedSampler` for JSONL data; all logs/checkpoints from rank 0 only
- **FineWeb streaming** (`StreamingHFDataset`): `--stream` flag streams any HF
  dataset without downloading; auto-shards across DDP ranks; packs tokenized text
  into fixed-length chunks; tested on `HuggingFaceFW/fineweb/sample-10BT`
- **Knowledge distillation** (`scripts/distill_aetherforge.py`): Qwen2.5-VL-7B
  teacher (4-bit NF4, ~6 GB) → AetherForge student (fp16); shared Qwen tokenizer
  so KL is in the same token space; teacher visual-token logits sliced to student
  vocab; loss `L = α·T²·KL + (1-α)·CE` with response-only masking; AMP,
  warmup+cosine LR, gradient accumulation, checkpoint resume, W&B, CSV + curve

### Fixes

- `ImportError: cannot import name 'SparseMoE'` — renamed to `AdaptiveMoE` with
  backwards-compat alias in `__init__.py`
- `AttributeError: 'Namespace' has no attribute 'use_wandb'` — `dest="use_wandb"`
  added to `--wandb` argparse arg
- `AttributeError: 'NoneType' has no attribute 'device'` in Qwen2.5-VL vision
  forward — `image_grid_thw` missing from dataset and training loop; fixed in both
- Distillation vocab mismatch: `tokenizer.vocab_size` (151643) vs actual max token
  ID (151664); fixed to `len(tokenizer)` = 151665
- Teacher head mismatch: Qwen2.5-VL model head = 152064 (includes visual tokens);
  sliced to student vocab before KL so visual tokens don't contaminate text loss

---

*Versioning follows [Semantic Versioning](https://semver.org).*
