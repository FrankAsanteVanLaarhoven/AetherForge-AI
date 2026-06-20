# AetherForge Model Card

**Version:** 0.3.0  
**Type:** Decoder-only causal language model  
**Licence:** MIT  
**Author:** Frank Asante Van Laarhoven — Newcastle University  
**Contact:** F.Van-Laarhoven2@newcastle.ac.uk  

---

## Model Description

AetherForge is a research transformer architecture designed to study efficient
attention, sparse expert routing, and differentiable tool use on consumer hardware.
Three checkpoints have been trained as part of the v0.2 development phase.

### Architecture Highlights

**MLAPlus** — Multi-Head Latent Attention with KV compression (DeepSeek-V2 MLA),
Rotary Position Embedding (RoPE, lazy freq cache, 1M+ context), and Flash Attention 2
(`F.scaled_dot_product_attention(is_causal=True)`).

**Coupled MoE (formerly "AQ-MoE")** — Sparse Mixture of Experts with a learned E×E
correlation matrix **J** in the router:
`s' = s + η·(s @ J)`. This lets the router learn which experts tend to co-activate.
Initialised to zero (standard top-k routing) and updated during training.
Load-balance loss: `L = E · Σ(f_i · P_i)`.

> **On "quantum-inspired" terminology:** Earlier documentation used terms like
> "Adaptive Quantum-Inspired MoE" and "entanglement scores". These are classical
> tensor operations — there is no quantum hardware or quantum computing involved.
> The coupling matrix J is simply a learned weight; "entanglement" refers to
> learned correlations between experts. The quantum naming is cosmetic and has
> been removed from the codebase.

**ForgeReasoningCore v2** — Gated iterative refinement across N steps with:
- 64 learned (key, value) memory slots read via soft attention with per-step gates
- `register_tool(step, fn)` — wire any Python callable into the forward pass

**Gradient checkpointing** — `model.enable_gradient_checkpointing()` halves VRAM
by recomputing activations during the backward pass.

### Model Variants

| Config | Parameters | d_model | Layers | Heads | Experts | moe_d_ff |
|--------|-----------|---------|--------|-------|---------|----------|
| 128M   | 126.5 M   | 512     | 6      | 8     | 8       | 2048     |
| 1B     | 728.6 M   | 2048    | 16     | 16    | 4       | 1024     |
| 7B     | ~7.0 B    | 4096    | 32     | 32    | 8       | 1792     |
| 13B    | ~13.0 B   | 5120    | 40     | 40    | 8       | 2048     |

> Note: MoE expert FFN size (`moe_d_ff`) is explicitly set to keep parameter counts
> in the intended range. Without this override `d_ff = 4·d_model` per expert,
> which would balloon the "1B" to ~13B actual parameters.

---

## Trained Checkpoints (v0.2)

### AetherForge 128M — FineWeb Pretrained

- **Checkpoint:** `outputs/aetherforge_fineweb_128M/final/model.pt`
- **Tokenizer:** Char-level (`ord(c) % 32000`), vocab size 32,000
- **Training data:** HuggingFace FineWeb `sample-10BT` (streaming, no download)
- **Steps:** 10,000 | **Seq len:** 256 | **Batch size:** 4 (×4 grad accum → effective 16)
- **Training loss (final):** ~1.5 nats LM loss (CSV values ×10 inflated — pre-logging-fix run)
- **Optimizer:** AdamW (fp32), lr 3 × 10⁻⁴, cosine decay
- **Training speed:** ~16,700 tok/s | ~13 min on RTX 4080 (16 GB)
- **VRAM (inference):** 0.51 GB

### AetherForge 128M — Qwen-Distilled

- **Checkpoint:** `outputs/aetherforge_distill_5k/final/model.pt`
- **Config:** `outputs/aetherforge_distill_5k/final/config.json` (vocab_size=151,665)
- **Tokenizer:** Qwen2.5-VL-7B-Instruct BPE tokenizer (151,665 vocab)
- **Teacher:** Qwen2.5-VL-7B-Instruct (4-bit NF4, loaded on same GPU)
- **Training data:** 3,000 Alpaca instruction-response pairs
- **Steps:** 5,000 | **Seq len:** 256 | **Temperature:** 3.0 | **Alpha:** 0.7
- **Distillation objective:** `L = 0.7 · T² · KL(p_teacher ‖ p_student) + 0.3 · CE`
- **Training speed:** ~284 tok/s | ~16 min on RTX 4080 (16 GB)
- **VRAM (inference):** 0.76 GB (larger embedding matrix; 187.8 M params total)

### AetherForge 1B — FineWeb Pretrained

- **Checkpoint:** `outputs/aetherforge_1B_pretrain/final/model.pt`
- **Tokenizer:** Char-level (`ord(c) % 32000`), vocab size 32,000
- **Training data:** HuggingFace FineWeb `sample-10BT` (streaming)
- **Config:** 728.6 M parameters (see variant table)
- **Steps:** 15,000 | **Seq len:** 256 | **Batch size:** 1 (×1 grad accum)
- **Final training loss:** 2.91 nats (step 15,000)
- **Optimizer:** AdamW8bit (bitsandbytes 0.49), lr 3 × 10⁻⁴, cosine decay
- **Training speed:** ~1,750 tok/s | ~60 min on RTX 4080 (16 GB)
- **VRAM (training):** ~8.95 GB (fp32 weights + 8-bit optimizer states)
- **VRAM (inference):** 2.91 GB

### AetherForge 1B — BPE Code Agent Fine-tune

- **Checkpoint:** `outputs/aetherforge_1B_code_agent/final/model.pt`
- **Config:** `outputs/aetherforge_1B_bpe/init/config.json` (vocab_size=151,665)
- **Tokenizer:** Qwen2.5-VL-7B-Instruct BPE tokenizer (151,665 vocab)
- **Base:** 1B pretrain adapted via `scripts/adapt_tokenizer.py` (embedding swapped,
  body weights preserved)
- **Training data:** 9,500 examples (5k CodeAlpaca + 1.5k ReAct templates +
  3k real execution traces)
- **Steps:** 5,000 | **Seq len:** 256 | **Batch size:** 1 (×8 grad accum)
- **Training precision:** bf16 (weights + activations) + gradient checkpointing +
  8-bit Adam — total peak VRAM 8.31 GB
- **Final training loss:** ~6.0 nats (response-masked CE on ReAct format)
- **Optimizer:** AdamW8bit, lr 3 × 10⁻⁵, cosine decay, 200 warmup steps
- **Training speed:** ~2,250 tok/s | ~75 min on RTX 4080 (16 GB)
- **VRAM (inference):** 3.89 GB (fp32 at load time)

---

## Evaluation Results (v0.2)

Evaluation performed on held-out data not seen during training.
See `scripts/eval_checkpoints.py` and `outputs/eval_results/eval_report.json`.

### Perplexity / CE Loss

| Model | Eval set | Tokenizer | CE (nats) | Perplexity | Steps | Notes |
|-------|----------|-----------|-----------|------------|-------|-------|
| 128M-base | 300 FineWeb chunks (skipped 15k) | char-level | 1.68 | **5.34** | 10k | — |
| 128M-distilled | 100 Alpaca examples (skipped 3k), response-only | Qwen BPE | 0.59 | **1.80** | 5k distil | Response tokens only |
| 1B-base | 300 FineWeb chunks (skipped 15k) | char-level | 2.52 | **12.39** | 15k | Undertrained (4k step eval) |
| 1B-base (final) | — | char-level | — | — | 15k | Final loss 2.91 nats (char-level PPL TBD) |
| 1B-BPE-code-agent | — | Qwen BPE | — | — | 5k FT | Fine-tune loss ~6.0; benchmark below |
| 128M-code-agent | — | Qwen BPE | — | — | 3k FT | Tool-use fine-tune |

**Important caveats:**
- Perplexity values across the char-level rows and the BPE row are **not comparable**.
  BPE tokens carry more information per step, so BPE ppl is inherently lower.
- 128M-distilled ppl=1.80 is response-only, on held-out Alpaca. It overstates
  generalisation ability — the model memorised the Alpaca format.
- 1B-base ppl=12.39 is **worse** than 128M-base (5.34). Root cause: the 1B saw
  ~3.8M tokens (15k steps × 1 batch × 256 seq) vs 128M's ~40M tokens (10k × 16
  effective batch × 256 seq). The 1B is severely undertrained relative to its size.

### Code Agent Benchmark

5-task benchmark run after fine-tuning (`scripts/agent_loop.py --benchmark`):

| Model | Pass@1 (5 tasks) | Tool calls generated | Notes |
|-------|-----------------|---------------------|-------|
| 128M-code-agent (ckpt-1500) | **0 / 5 (0%)** | 0 | Generates "." × 150 tokens |
| 128M-distilled (base) | 0 / 5 (0%) | 0 | Same degenerate pattern |
| 1B-BPE-code-agent (step-5k) | **0 / 5 (0%)** | 0 | Incoherent token spacing; see root cause |

**Root cause — 128M (0% pass rate):** Repeated period tokens from degenerate distribution.
Causes: only 6,500 fine-tune examples → overfitting; no SFT warmup before ReAct format.

**Root cause — 1B-BPE (0% pass rate):** Model generates incoherent whitespace-scattered
characters. Observed output: `"  ink  in  2  in  ;  2  ))  ,  "  ...  def  =  in  print`.
This is fundamentally different from the 128M failure (repeated tokens vs. incoherent
tokens), and reveals a more tractable problem:
1. **Random embedding init**: The char→BPE vocabulary adapter randomly initialises all
   151,665 BPE embedding rows. The model has never seen BPE tokens — every ID is new.
2. **Insufficient fine-tune data for BPE adaptation**: 5,000 steps × 8 effective batch
   × 256 tokens = 10.2M training tokens. At 151k vocabulary, the embedding needs far
   more exposure per token to learn meaningful representations.
3. **Char-level body weights + BPE surface**: The attention, MoE, and FRC layers encode
   char-level patterns. BPE compound tokens (e.g. `def`, `print(`, `True`) map to
   different IDs than any char the pretrained body has seen.
4. **Training loss plateau at ~6.0 nats**: Random baseline = ln(151,665) ≈ 11.9 nats;
   achieved 6.0 nats = perplexity ~403 → learning but far from coherent generation
   (a well-trained 1B would reach < 2.5 nats on code).

**What is working:**
- The infrastructure is fully correct: real tool execution (subprocess), agent loop,
  `TOOL_CALL:`/`OBSERVATION:` injection, benchmark harness
- 8.31 GB VRAM fine-tune proves bf16+GC+8-bit-Adam enables 1B training on 16 GB GPU
- The tokenizer adapter correctly preserves body weights

**Path to a functional code agent:**
The fundamental issue is the char→BPE vocabulary gap. Two routes:
- **Route A (less compute):** Start from a pretrained BPE model (e.g., Llama-3.2-1B
  or Qwen2.5-0.5B) and fine-tune with the same code-agent pipeline. Body weights
  already encode BPE token semantics.
- **Route B (more data):** Pretrain the 1B model from scratch on a BPE-tokenized
  corpus (FineWeb re-tokenised with Qwen tokenizer) for 100k+ steps, then fine-tune.
  This aligns body weights with BPE IDs.

### Generation Quality

Qualitative samples (temperature=0.7, top_p=0.9, 80 new tokens, 5 prompts):

| Model | Generation pattern | Instruction following |
|-------|-------------------|----------------------|
| 128M-base | Word/phrase salad with English structure (FineWeb LM) | None — LM only |
| 128M-distilled | Repeated punctuation / `of of of of` | None — degenerate |
| 1B-base | Garbled character sequences | None — undertrained |
| 128M-code-agent | Repeated `.` tokens | None — degenerate |

### Hardware Budget

| Task | VRAM | Notes |
|------|------|-------|
| 128M inference | 0.51 GB | Char-level tokenizer |
| 128M distilled inference | 0.76 GB | Qwen BPE embedding (151 k vocab) |
| 1B inference | 2.91 GB | fp32 char-level |
| 1B BPE inference | 3.89 GB | fp32, 151k vocab |
| 1B BPE fine-tune (bf16+GC+8-bit Adam) | **8.31 GB** | bf16 weights+activations |
| Distillation (teacher + 128M student) | ~9.25 GB | Qwen 4-bit + student fp32 |
| Pretraining 128M (AMP, seq=256) | ~1.10 GB | Large batch possible |
| Pretraining 1B + gc + 8-bit Adam | ~8.95 GB | RTX 4080 16 GB + Isaac Sim running |
| Qwen2.5-VL fine-tuning (4-bit + LoRA) | 11–14 GB | seq=256 |

---

## Training Configuration

```bash
# 128M on FineWeb (10k steps, char-level)
conda run -n ml-torch python scripts/train_aetherforge.py \
    --stream --config 128M --steps 10000 --seq-len 256

# 128M distilled from Qwen2.5-VL-7B (5k steps)
conda run -n ml-torch python scripts/distill_aetherforge.py \
    --config 128M --temperature 3.0 --alpha 0.7 --steps 5000

# 1B on FineWeb (gradient checkpointing + 8-bit Adam for 16 GB GPU)
conda run -n ml-torch python scripts/train_aetherforge.py \
    --stream --config 1B --gradient-checkpointing --8bit-adam \
    --steps 15000 --seq-len 256 --batch-size 1

# Swap 1B char-level vocab → Qwen BPE (151k tokens)
conda run -n ml-torch python scripts/adapt_tokenizer.py \
    --checkpoint outputs/aetherforge_1B_pretrain/final/model.pt \
    --config 1B --output outputs/aetherforge_1B_bpe/init

# Fine-tune 1B-BPE on code + execution traces + ReAct tool-use
# bf16 weights + gradient checkpointing + 8-bit Adam = 8.31 GB VRAM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n ml-torch python scripts/finetune_code_agent.py \
    --checkpoint outputs/aetherforge_1B_bpe/init/model.pt \
    --config     outputs/aetherforge_1B_bpe/init/config.json \
    --data       data/code_agent_data.jsonl \
    --output     outputs/aetherforge_1B_code_agent \
    --steps 5000 --lr 3e-5 --batch-size 1 --grad-accum 8 \
    --max-length 256 --warmup-steps 200 \
    --8bit-adam --gradient-checkpointing --amp

# Run evaluation
conda run -n ml-torch python scripts/eval_checkpoints.py \
    --n-chunks 300 --n-alpaca 100 --output outputs/eval_results

# Run code agent benchmark
conda run -n ml-torch python scripts/agent_loop.py \
    --checkpoint outputs/aetherforge_1B_code_agent/final/model.pt \
    --config     outputs/aetherforge_1B_bpe/init/config.json \
    --benchmark
```

Training objectives:
```
Pretraining:   L = L_LM + α · L_balance        (α = 0.01)
Distillation:  L = α · T² · KL(p_t ‖ p_s) + (1-α) · CE_LM
                                                (α = 0.7, T = 3.0)
```

---

## Intended Uses

- Architecture research on attention (MLAPlus), MoE (AQ-MoE), and memory mechanisms (FRC)
- Knowledge distillation: train AetherForge as a student from Qwen2.5-VL-7B
- FineWeb / DCLM streaming pretraining on a single RTX 4080 or similar consumer GPU
- Custom task-specific pretraining and fine-tuning

### Out of Scope

- Production deployment as a general-purpose assistant without additional alignment and
  significantly more pretraining data
- Safety-critical applications without additional RLHF/DPO alignment

---

## Known Limitations

- **Char-level tokenizer (128M, 1B base):** efficient for prototyping but produces
  poor instruction-following. A BPE tokenizer (or the Qwen tokenizer) is required for
  real language-model evaluation and comparison.
- **Distilled 128M — limited generalisation:** 3k Alpaca examples produce near-zero
  held-out CE on the same distribution but degenerate freeform generation. 10k–50k
  diverse instruction examples are needed for reliable instruction following.
- **1B in-progress:** checkpoint at ~4k of 15k steps; loss still converging. Final
  perplexity TBD.
- **AQ-MoE coupling matrix and FRC memory slots** are novel components that have not
  been ablated at scale; contribution vs. standard MoE/attention is unquantified.
- **DDP** tested on single-node; cross-node training requires `MASTER_ADDR`/`MASTER_PORT`.

---

## Citation

```bibtex
@software{aetherforge2026,
  author    = {Van Laarhoven, Frank Asante},
  title     = {AetherForge-AI: A Hybrid Latent-Entangled Transformer and
               Efficient Fine-Tuning Toolkit},
  year      = {2026},
  version   = {0.2.0},
  url       = {https://github.com/FrankAsanteVanLaarhoven/AetherForge-AI},
  note      = {Newcastle University},
}
```

---

## Repository

`github.com/FrankAsanteVanLaarhoven/AetherForge-AI`

Scripts:
- `scripts/train_aetherforge.py` — pretraining (single-GPU + DDP + FineWeb streaming + 8-bit Adam)
- `scripts/distill_aetherforge.py` — knowledge distillation from Qwen2.5-VL-7B
- `scripts/eval_checkpoints.py` — evaluate all trained checkpoints (perplexity + generation)
- `scripts/generate_synthetic_data.py` — download Alpaca data for distillation
- `scripts/finetune_qwen25_vl.py` — Qwen2.5-VL LoRA fine-tuning
- `scripts/serve.py` — FastAPI inference server
- `scripts/evaluate_model.py` — Qwen2.5-VL benchmarks
- `Training_Dashboard.ipynb` — 13-panel monitoring dashboard
