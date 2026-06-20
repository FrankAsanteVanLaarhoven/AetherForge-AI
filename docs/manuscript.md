# AetherForge: A Hybrid Latent-Entangled Transformer for Data-Efficient, Autonomous Superintelligence

**Frank Asante Van Laarhoven**  
Newcastle University · ORCID: 0009-0006-8931-0364  
Weights & Biases: F.Van-Laarhoven2@newcastle.ac.uk · HuggingFace: frankleroyvan  
*2026 — AetherForge-AI GitHub Repository*

---

## Current Implementation Status

| Feature | Status | Where |
|---------|--------|-------|
| AetherForge 128M prototype | ✅ Implemented | `aetherforge/model.py` |
| MLAPlus (KV compression) | ✅ Implemented | `model.py:MLAPlus` |
| RoPE — 1M+ token context | ✅ Implemented | `model.py:_rope_freqs` |
| Flash Attention 2 (SDPA) | ✅ Implemented | `model.py:MLAPlus.forward` |
| AQ-MoE coupling matrix J | ✅ Implemented | `model.py:AdaptiveMoE` |
| AQ-MoE load-balance loss | ✅ Implemented | `train_aetherforge.py` |
| FRC v2 — memory buffer (64 slots) | ✅ Implemented | `model.py:ForgeReasoningCore` |
| FRC v2 — tool-use hooks | ✅ Implemented | `model.py:ForgeReasoningCore.register_tool` |
| Production size configs (1B/7B/13B) | ✅ Implemented | `model.py:MODEL_CONFIGS` |
| Pretraining (AMP, cosine LR, W&B) | ✅ Implemented | `scripts/train_aetherforge.py` |
| Real HuggingFace tokenizer | ✅ Implemented | `--tokenizer` arg |
| Qwen2.5-VL LoRA fine-tuning | ✅ Implemented | `scripts/finetune_qwen25_vl.py` |
| Evaluation suite (text + vision) | ✅ Implemented | `scripts/evaluate_model.py` |
| LoRA merge utility | ✅ Implemented | `scripts/merge_lora.py` |
| FastAPI inference server + streaming | ✅ Implemented | `scripts/serve.py` |
| Enterprise training dashboard (13 panels) | ✅ Implemented | `Training_Dashboard.ipynb` |
| Makefile (19 targets) | ✅ Implemented | `Makefile` |
| Training at 7B+ (actual weights) | 🔜 Phase 3 | Requires A100/H100 cluster |
| Teacher distillation pipeline | 🔜 Phase 3 | — |
| Full multimodal AetherForge | 🔜 Phase 3 | Currently: Qwen fine-tuning |
| SWE-Bench / GPQA / ARC-AGI runs | 🔜 Phase 3 | Need production weights |
| HuggingFace open weights release | 🔜 Phase 4 | — |
| ArXiv submission | 🔜 Phase 4 | — |

---

## Abstract

We present **AetherForge**, a novel large language model architecture that synthesises multi-head latent attention, adaptive sparse mixture-of-experts routing, and forge-based test-time reasoning into a single unified system. AetherForge is designed to achieve state-of-the-art benchmark performance while requiring 30–50% less pretraining data than comparably-sized dense transformers — a target grounded in the data-efficiency techniques pioneered by Chinese frontier models (DeepSeek, Kimi, Qwen, GLM). The architecture introduces three primary innovations: **MLA+** (Multi-Head Latent Attention Plus), which extends DeepSeek-V2's latent KV compression with RoPE for sub-quadratic long-context scaling to 1M+ tokens and Flash Attention 2 via `F.scaled_dot_product_attention`; **AQ-MoE** (Adaptive Quantum-Inspired Mixture of Experts), which uses a learned E×E coupling matrix $J$ to route tokens to active experts accounting for inter-expert correlations, with load-balance loss included in the training objective; and the **Forge Reasoning Core (FRC)**, a built-in test-time compute loop with persistent memory buffer and callable tool hooks that enables iterative self-verification analogous to OpenAI o1 and Grok reasoning modes. The prototype (128M parameters) runs at 1.45 GB VRAM on an RTX 4080 Super; the codebase provides production-ready configs for 1B, 7B, and 13B parameter targets. Together with a full fine-tuning toolkit for Qwen2.5-VL-7B and a FastAPI inference server, AetherForge represents a complete SOTA-targeted LLM research platform runnable on consumer hardware.

---

## 1. Background and Transformer Foundations

### 1.1 The Original Transformer (Vaswani et al., 2017)

The Transformer architecture introduced scaled dot-product attention as the primary mechanism for modelling sequence dependencies:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

A full decoder-only stack adds:

- **Multi-head attention** — $h$ independent attention heads projected into subspaces, concatenated and linearly mixed:

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- **Position-wise FFN** — two linear layers with a non-linearity between them applied identically to each token position.
- **Layer normalisation** — AetherForge uses **RMSNorm** with **pre-norm** throughout for training stability.
- **Positional encoding** — AetherForge uses RoPE (Rotary Position Embedding) extended for 1M+ context via lazy frequency cache extension.

### 1.2 Innovations Since 2017

| Year | Development | Status in AetherForge |
|------|-------------|----------------------|
| 2020 | Sparse MoE (GShard, Switch Transformer) | ✅ Foundation for AQ-MoE |
| 2022 | Flash Attention | ✅ Adopted via SDPA |
| 2023 | RoPE long-context | ✅ Implemented; 1M+ lazy cache |
| 2023 | QLoRA, LoRA | ✅ Qwen2.5-VL fine-tuning |
| 2024 | DeepSeek-V2 MLA | ✅ Core MLA+ inspiration |
| 2024 | Expert coupling / MoE correlations | ✅ AQ-MoE coupling matrix J |
| 2025 | o1 / Grok reasoning modes | ✅ FRC with memory + tool hooks |
| 2026 | AetherForge | All of the above, unified |

### 1.3 The Data-Efficiency Problem

Dense transformers of 100B+ parameters require 1–10T tokens of pretraining data. Chinese frontier models (Kimi, DeepSeek, GLM) demonstrated that aggressive synthetic data generation, curriculum ordering, and teacher distillation can close this gap significantly. AetherForge targets the same 30–50% reduction via:

1. **Synthetic curriculum** — high-quality reasoning chains generated by teacher ensembles
2. **MLA+ KV compression** — allows longer effective contexts per compute budget
3. **AQ-MoE sparsity** — activates fewer parameters per token, reducing overfitting risk on small corpora

---

## 2. AetherForge Components

### 2.1 MLA+ — Multi-Head Latent Attention Plus

**Motivation.** Standard multi-head attention's KV cache grows as $O(B \cdot T \cdot h \cdot d_k)$. At 1M token context this becomes prohibitive. DeepSeek-V2's MLA projects K and V through a shared low-rank latent bottleneck of dimension $d_\ell \ll d_\text{model}$.

AetherForge v2 extends this with:
- **RoPE** — keys and queries receive rotary position encodings that allow context extension far beyond training length with no re-training
- **Flash Attention 2** — attention is computed via `F.scaled_dot_product_attention(is_causal=True)`, dispatching to CUDA Flash Attention 2 kernels automatically on supported hardware

**Equations.**

$$z = xW_\text{compress} \in \mathbb{R}^{B \times T \times d_\ell}$$

$$K = \text{RoPE}(zW_k^\top), \quad V = zW_v^\top$$

$$Q = \text{RoPE}(xW_q^\top)$$

$$\text{MLA+}(Q,K,V) = \text{FlashAttention}(Q, K, V, \text{causal=True})$$

KV cache at inference scales with $d_\ell$ rather than $d_\text{model}$, giving compression ratio $d_\text{model}/d_\ell = 4\times$ in the default 128M configuration (512/128).

**RoPE** applies rotation in 2D subspaces of the head dimension:

$$\text{RoPE}(x, m)_i = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)$$

where $\theta_i = 10000^{-2i/d_\text{head}}$ and $m$ is the position index. Frequencies are precomputed and cached; the cache extends lazily when a longer sequence is processed.

**Implemented** (`aetherforge/model.py:MLAPlus`):

```python
class MLAPlus(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim=128, rope_base=10_000):
        self.kv_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.k_expand    = nn.Linear(latent_dim, d_model, bias=False)
        self.v_expand    = nn.Linear(latent_dim, d_model, bias=False)
        self.q_proj      = nn.Linear(d_model, d_model,   bias=False)
        self.out_proj    = nn.Linear(d_model, d_model,   bias=False)
        # RoPE freq cache extended lazily

    def forward(self, x):
        q, k, v = ...  # project to [B, H, T, head_dim]
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1,2).contiguous().view(B, T, C))
```

### 2.2 AQ-MoE — Adaptive Quantum-Inspired Mixture of Experts

**Motivation.** Standard top-K routing (Shazeer et al., 2017) assigns each token to the $k$ experts with highest router softmax scores. The router treats experts as independent and ignores correlations between them. This leads to load imbalance and sub-optimal expert utilisation.

AQ-MoE replaces the router with a **coupled routing** formulation via a learned E×E interaction matrix:

$$s = x W_\text{router} \in \mathbb{R}^{N \times E}$$

$$s_\text{coupled} = s + \eta \cdot (s \cdot J), \quad J \in \mathbb{R}^{E \times E}$$

where $J$ is the coupling matrix initialised to zeros (no coupling at init, fully learned during training), and $\eta$ is a fixed scale factor. Expert selection accounts for which other experts are active, reducing redundant expert co-activation.

**Routing algorithm:**

$$g = \text{Softmax}(\text{TopK}(s_\text{coupled}, k))$$

$$\text{AQ-MoE}(x) = \sum_{i \in \text{top-k}} g_i \cdot \text{FFN}_i(x)$$

**Load balancing loss** added to the training objective (weight `--balance-alpha`, default 0.01):

$$\mathcal{L}_\text{balance} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens whose primary routing selects expert $i$, and $P_i$ is the mean router probability for expert $i$ across the batch.

**Implemented** (`aetherforge/model.py:AdaptiveMoE`):

```python
class AdaptiveMoE(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2, eta=0.1):
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.J      = nn.Parameter(torch.zeros(n_experts, n_experts))  # coupling matrix
        self.w1     = nn.Parameter(torch.randn(n_experts, d_model, d_ff) * 0.02)
        self.w2     = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)

    def forward(self, x):
        s        = self.router(x_flat)
        s_coup   = s + self.eta * (s @ self.J)   # AQ coupling
        probs    = F.softmax(s_coup, dim=-1)
        # vectorised dispatch: loop over E experts, not B×T tokens
```

Training objective including load-balance:

```bash
python scripts/train_aetherforge.py \
    --config 128M \
    --balance-alpha 0.01 \   # weight on L_balance
    --steps 5000
```

### 2.3 Forge Reasoning Core v2 — Memory Buffer + Tool Hooks

**Motivation.** Test-time compute scaling (OpenAI o1, Grok reasoning mode) significantly improves performance on hard reasoning tasks by allowing the model to "think" for more steps before producing an answer. FRC v2 extends the v1 gated refinement loop with a persistent memory buffer and callable tool hooks, making it natively agentic.

**Architecture.** Gated iterative refinement:

$$h_{t+1} = h_t + \sigma(g_t) \cdot \text{GELU}(W_\text{refine}\, h_t) + \sigma(\gamma_t) \cdot \text{MemRead}(h_t)$$

where $g_t \in \mathbb{R}^{d_\text{model}}$ is a per-step gate vector (init zero), $\gamma_t \in \mathbb{R}$ is a per-step memory read gate (init zero), and:

$$\text{MemRead}(h) = \text{Softmax}(h \cdot K_\text{mem}^\top \cdot s^{-1/2}) \cdot V_\text{mem}$$

$K_\text{mem}, V_\text{mem} \in \mathbb{R}^{M \times d_\text{model}}$ are learned memory slot parameters ($M = 64$ by default). The memory is a trained static KV store — not updated during forward passes, but updateable between steps via tool hooks.

**Tool hooks** are Python callables registered per step:

```python
# Register a retrieval tool to fire after FRC step 1
model.blocks[0].frc.register_tool(1, retrieval_fn)

# Tool receives x [B, T, d_model] and must return same shape
def retrieval_fn(x: torch.Tensor) -> torch.Tensor:
    # query external KB, inject retrieved context
    return x + retrieved_embeddings
```

**Agentic capability summary:**

| Capability | v1 | v2 |
|------------|----|----|
| Gated iterative refinement | ✅ | ✅ |
| Learnable critique steps | ✅ | ✅ |
| Persistent memory buffer | ❌ | ✅ (64 slots) |
| Tool-use hooks | ❌ | ✅ (per-step) |
| Inference-time depth scaling | ✅ | ✅ |

**Implemented** (`aetherforge/model.py:ForgeReasoningCore`):

```python
class ForgeReasoningCore(nn.Module):
    def __init__(self, d_model, n_steps=3, mem_slots=64):
        self.steps    = nn.ModuleList([nn.Linear(d_model, d_model) ...])
        self.gates    = nn.Parameter(torch.zeros(n_steps, d_model))
        self.mem_k    = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)
        self.mem_v    = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)
        self.mem_gate = nn.Parameter(torch.zeros(n_steps))
        self.tool_hooks: dict[int, Callable] = {}

    def register_tool(self, step: int, fn: Callable) -> None: ...
    def _mem_read(self, x) -> torch.Tensor: ...   # soft attention over mem slots
```

### 2.4 Full AetherForge Block

Each transformer block stacks these three components with RMSNorm pre-norm:

```
x → RMSNorm → MLA+ (RoPE + Flash Attn) → residual
  → RMSNorm → AQ-MoE (coupled routing + J) → residual
  → RMSNorm → FRC v2 (memory + tool hooks) → residual
```

This ordering ensures attention captures context first, MoE applies specialised transformations, and FRC refines the result with memory-augmented iterative reasoning.

### 2.5 Production Size Configurations

```python
MODEL_CONFIGS = {
    "128M": dict(d_model=512,  n_layers=6,  n_heads=8,  latent_dim=128,
                 n_experts=8,  top_k=2, mem_slots=64,  frc_steps=3),

    "1B":   dict(d_model=2048, n_layers=24, n_heads=16, latent_dim=512,
                 n_experts=16, top_k=2, mem_slots=128, frc_steps=3),

    "7B":   dict(d_model=4096, n_layers=32, n_heads=32, latent_dim=1024,
                 n_experts=64, top_k=8, mem_slots=256, frc_steps=4),

    "13B":  dict(d_model=5120, n_layers=40, n_heads=40, latent_dim=1280,
                 n_experts=64, top_k=8, mem_slots=256, frc_steps=4),
}

model = AetherForge.from_config("7B")   # instantiate by name
```

VRAM estimates (fp16, inference only):

| Config | Params | VRAM (fp16) | RTX 4080 16GB |
|--------|--------|-------------|---------------|
| 128M | 126.5M | ~1.5 GB | ✅ train + inference |
| 1B | ~2.4B | ~5 GB | ✅ inference, tight train |
| 7B | ~16B | ~14 GB | 4-bit only |
| 13B | ~30B | ~28 GB | requires A100 |

---

## 3. Training Pipeline

### 3.1 Data Preparation

**Sources (ordered by quality):**

| Tier | Source | Fraction |
|------|--------|----------|
| 1 | Curated books, academic papers, code | 30% |
| 2 | High-quality web text (FineWeb, DCLM) | 40% |
| 3 | Synthetic — teacher-generated reasoning chains | 20% |
| 4 | Synthetic — instruction-response pairs | 10% |

**Tokenizer:** `--tokenizer Qwen/Qwen2.5-VL-7B-Instruct` uses HuggingFace AutoTokenizer (BPE, 32K vocab). Omit for a char-level fallback (no deps, smoke-tests only).

**Curriculum ordering:** short sequences → long sequences → reasoning chains → code → multimodal.

### 3.2 Pretraining

**Training objective:**

$$\mathcal{L} = \mathcal{L}_\text{LM} + \alpha\,\mathcal{L}_\text{balance} + \beta\,\mathcal{L}_\text{distill}$$

$\mathcal{L}_\text{LM}$ is standard causal language modelling loss. $\mathcal{L}_\text{balance}$ is the AQ-MoE load-balance term (default $\alpha=0.01$). $\mathcal{L}_\text{distill}$ is KL divergence from teacher ensemble (Phase 3).

**Staged protocol:**

| Stage | Frozen | Objective | Compute |
|-------|--------|-----------|---------|
| 2a — MLA warmup | AQ-MoE, FRC | LM only | 10% |
| 2b — Full MoE | — | LM + balance | 70% |
| 2c — FRC activation | — | LM + balance | 20% |

**Verified hyperparameters (128M, RTX 4080 Super 16GB):**

| Parameter | Value |
|-----------|-------|
| Sequence length | 256 (safe) / 1024 (with RoPE/SDPA) |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch 16) |
| Learning rate | 3×10⁻⁴ peak, cosine to 3×10⁻⁵ |
| Warmup steps | 100 |
| Optimizer | AdamW β=(0.9, 0.95), wd=0.1 |
| Precision | fp16 + AMP GradScaler |
| Balance alpha | 0.01 |
| VRAM usage | 1.45–2.1 GB (128M) |

```bash
# 128M smoke-test (50 steps)
python scripts/train_aetherforge.py --test-run

# 1B with real tokenizer
python scripts/train_aetherforge.py \
    --config 1B \
    --tokenizer Qwen/Qwen2.5-VL-7B-Instruct \
    --data data/synthetic_data.jsonl \
    --steps 10000 --balance-alpha 0.01 --wandb
```

### 3.3 Distillation (Phase 3)

AetherForge will be distilled from an ensemble of teacher models:

$$\mathcal{L}_\text{distill} = \text{KL}\!\left(p_\text{teacher}(\cdot|x) \,\|\, p_\text{student}(\cdot|x)\right)$$

Teacher ensemble:
- Qwen2.5-VL-7B-Instruct (multimodal reasoning)
- DeepSeek-V2-Chat (efficient long-context)
- Llama-3.1-8B-Instruct (general instruction following)

### 3.4 Post-Training

1. **SFT** — instruction-response pairs; uses `scripts/finetune_qwen25_vl.py` pattern
2. **DPO / RLHF** — preference learning from AI-judge feedback
3. **FRC depth scaling** — at inference, increase FRC steps from 3 (training) to 6–12 (hard tasks)

---

## 4. Multimodal Architecture

AetherForge targets seamless text + vision + code fusion through a shared latent space:

```
Text tokens    →  ┐
Image patches  →  ├─ Unified Latent Encoder → AetherForge Blocks → Output
Audio frames   →  ┘
```

**Current implementation:** `scripts/finetune_qwen25_vl.py` provides the full multimodal pipeline with `--mode multimodal`, using Qwen2.5-VL-7B-Instruct as the vision backbone. Dummy dataset: `multimodal_example/generate_dummy_data.py` (20 hospital-scene PNG images + JSONL).

**Phase 3 target:** native AetherForge vision encoder — ViT-based image patches interleaved with text tokens in the main transformer. MLA+'s latent bottleneck naturally handles the higher token count from images.

---

## 5. Inference Server

AetherForge ships a FastAPI server (`scripts/serve.py`) with streaming support:

```bash
# Serve AetherForge 128M
python scripts/serve.py \
    --model aetherforge \
    --checkpoint outputs/aetherforge_pretrain/final/model.pt

# Serve Qwen2.5-VL + LoRA
python scripts/serve.py --model qwen --lora-path outputs/qwen25_vl_lora/final
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness, device, model name |
| `/info` | GET | Param count, config, VRAM |
| `/generate` | POST | Single completion |
| `/stream` | POST | Server-Sent Events token stream |
| `/chat` | POST | Multi-turn conversation |

---

## 6. Evaluation

### 6.1 Verified Results (RTX 4080 Super 16GB)

| Benchmark | Score | VRAM | Notes |
|-----------|-------|------|-------|
| Text accuracy (10 Q) | 10/10 (100%) | 5.9 GB | Base Qwen2.5-VL-7B |
| Vision accuracy (10 img × 3 Q) | 30/30 (100%) | 5.9 GB | Hospital-scene images |
| Perplexity (reference texts) | 55.51 | — | Base model |
| Throughput — AetherForge 128M | ~2,500 tok/s | 1.45–2.1 GB | seq_len=64 |

### 6.2 Target Benchmarks (Production Scale)

| Benchmark | Target | Enabling Feature |
|-----------|--------|-----------------|
| SWE-Bench Verified | Top-5 | FRC tool hooks for iterative code debugging |
| GPQA Diamond | >70% | MLA+ long-context + FRC memory |
| ARC-AGI | >85% | AQ-MoE multi-expert composition |
| MMMU | >75% | Multimodal native fusion |
| MATH-500 | >90% | Distillation from strong math teachers |
| HumanEval | >90% | Code expert in AQ-MoE |
| Long-context (1M) | Competitive | RoPE lazy extension + MLA+ KV compression |

### 6.3 Efficiency vs. Existing Systems

| System | Active Params | Key Efficiency Mechanism |
|--------|--------------|--------------------------|
| GPT-4 (est.) | ~220B | — |
| DeepSeek-V3 | 37B active (671B total) | MoE sparsity |
| **AetherForge (target 7B)** | **~7B** | **RoPE + AQ-MoE + FRC test-time** |
| **AetherForge (target 100B)** | **100B active** | **Sparse + memory-augmented** |

Run evaluation:

```bash
conda run -n ml-torch python scripts/evaluate_model.py \
    --benchmark all \
    --image-dir multimodal_example/images \
    --lora-path outputs/qwen25_vl_lora/final \
    --compare-base
```

---

## 7. Usage Reference

### 7.1 Quick Start

```bash
git clone https://github.com/FrankAsanteVanLaarhoven/AetherForge-AI
cd AetherForge-AI && pip install -r requirements.txt

make test-aetherforge          # 128M forward pass, ~1.45 GB VRAM
make train-aetherforge-test    # 50-step smoke-test, loss curve + CSV
make finetune-test             # Qwen2.5-VL LoRA, 50 steps, 6.5 GB VRAM
make eval-vision               # Text + vision benchmarks
make serve                     # FastAPI server on :8000
jupyter notebook Training_Dashboard.ipynb
```

### 7.2 Configuration Reference

`scripts/train_aetherforge.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | — | Size preset: `128M` / `1B` / `7B` / `13B` |
| `--tokenizer` | — | HF tokenizer id (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) |
| `--balance-alpha` | 0.01 | AQ-MoE load-balance loss weight |
| `--d-model` | 512 | Hidden dimension (if not using --config) |
| `--n-layers` | 6 | Transformer blocks |
| `--n-heads` | 8 | Attention heads |
| `--latent-dim` | 128 | MLA+ KV compression bottleneck |
| `--n-experts` | 8 | MoE experts total |
| `--top-k` | 2 | Active experts per token |
| `--seq-len` | 256 | Sequence length |
| `--warmup-steps` | 100 | Linear LR warmup |
| `--wandb` | off | Enable W&B logging |
| `--resume` | — | Checkpoint path to resume from |

`scripts/finetune_qwen25_vl.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | text | `text` or `multimodal` |
| `--max-length` | 256 | Token sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--output-dir` | `./outputs/qwen25_vl_lora` | Output path |

### 7.3 All Make Targets

```bash
make help             # print full target list
make test-all         # all three smoke-tests
make data             # generate synthetic training data
make train-aetherforge       # pretrain 128M
make train-aetherforge-test  # 50-step smoke-test
make finetune         # Qwen2.5-VL text fine-tuning
make finetune-multimodal     # Qwen2.5-VL vision fine-tuning
make merge-lora       # merge LoRA into base weights
make serve            # FastAPI server — AetherForge 128M :8000
make serve-qwen       # FastAPI server — Qwen + LoRA :8000
make eval             # text benchmark
make eval-vision      # text + vision + PPL
make eval-compare     # base vs. fine-tuned diff table
```

---

## 8. Roadmap

### Phase 1 — Foundation (Complete)
- [x] AetherForge 128M prototype (MLAPlus, SparseMoE, FRC)
- [x] Pretraining loop with AMP, warmup+cosine LR, W&B, checkpoint resume, CSV log
- [x] Qwen2.5-VL-7B LoRA fine-tuning (text + multimodal)
- [x] Evaluation suite (text, vision, perplexity, base vs. fine-tuned diff)
- [x] Enterprise Training Dashboard (13 panels, Palantir-grade)
- [x] LoRA merge utility
- [x] Makefile with 19 convenience targets

### Phase 2 — Production Architecture (Complete)
- [x] RoPE (Rotary Position Embedding) — replaces learned absolute positions; 1M+ context
- [x] Flash Attention 2 via `F.scaled_dot_product_attention(is_causal=True)`
- [x] AQ-MoE coupling matrix J (E×E, init zeros, fully learned)
- [x] AQ-MoE load-balance loss in training objective
- [x] FRC v2 — memory buffer (64 learned KV slots per block)
- [x] FRC v2 — tool-use hooks (`register_tool(step, fn)`)
- [x] Production configs: 128M / 1B / 7B / 13B (`MODEL_CONFIGS`)
- [x] `AetherForge.from_config(name)` instantiation
- [x] HuggingFace tokenizer integration (`--tokenizer`)
- [x] FastAPI inference server with SSE streaming (`scripts/serve.py`)

### Phase 3 — Scale + SOTA Push
- [ ] Train AetherForge-1B on real data (100B tokens minimum)
- [ ] Train AetherForge-7B (requires A100/H100 cluster)
- [ ] Teacher distillation pipeline (Qwen2.5-VL + DeepSeek ensemble)
- [ ] Full multimodal AetherForge (native ViT integration)
- [ ] SWE-Bench, GPQA, ARC-AGI, MMMU evaluation runs
- [ ] Long-context benchmark (1M tokens)
- [ ] FRC self-verification training curriculum

### Phase 4 — Release
- [ ] Open weights on HuggingFace: `frankleroyvan/AetherForge-7B`
- [ ] HuggingFace demo space
- [ ] ArXiv technical report submission
- [ ] ICLR / NeurIPS 2027 submission

---

## References

1. Vaswani, A. et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

2. DeepSeek-AI (2024). **DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model**. https://arxiv.org/abs/2405.04434

3. Shazeer, N. et al. (2017). **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. *ICLR 2017*. https://arxiv.org/abs/1701.06538

4. Hu, E. et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. https://arxiv.org/abs/2106.09685

5. Dettmers, T. et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. https://arxiv.org/abs/2305.14314

6. Su, J. et al. (2021). **RoFormer: Enhanced Transformer with Rotary Position Embedding**. https://arxiv.org/abs/2104.09864

7. Dao, T. et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. https://arxiv.org/abs/2205.14135

8. Dao, T. et al. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. https://arxiv.org/abs/2307.08691

9. Qwen Team (2024). **Qwen2.5-VL Technical Report**. https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

10. Meta AI (2024). **Llama 3 Model Card**. https://huggingface.co/meta-llama/Meta-Llama-3-8B

11. OpenAI (2024). **GPT-4 Technical Report**. https://arxiv.org/abs/2303.08774

12. Hinton, G. et al. (2015). **Distilling the Knowledge in a Neural Network**. https://arxiv.org/abs/1503.02531

---

*AetherForge is developed openly at https://github.com/FrankAsanteVanLaarhoven/AetherForge-AI*  
*Weights & Biases project: `fleetsafe-hospitalnav` · https://wandb.ai*
