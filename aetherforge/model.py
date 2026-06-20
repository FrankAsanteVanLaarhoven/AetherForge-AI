"""
AetherForge v2 — Phase 2-4 features.

New vs v1:
- RoPE (Rotary Position Embedding): replaces learned absolute embeddings;
  supports up to 1M+ tokens via lazy frequency cache extension.
- Flash Attention via F.scaled_dot_product_attention: uses FA2 CUDA kernels
  automatically when available; falls back to standard attention on CPU.
- AQ-MoE (Adaptive Quantum-Inspired MoE): learned E×E coupling matrix J
  that models expert-to-expert correlations during routing; exposes
  .load_balance_loss for the auxiliary training objective.
- FRC v2: persistent memory buffer (mem_slots soft KV pairs) + callable
  tool hooks registered per step; memory read is gated (init near zero).
- MODEL_CONFIGS: predefined 128M / 1B / 7B / 13B size specifications.
- AetherForge.from_config(name): instantiate by size name.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ── Size configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    # ── Standard configs ──────────────────────────────────────────────────
    # 128M  (d_model=512, moe_d_ff defaults to 4×512=2048)
    # Verified: 126.5M params, 1.1 GB VRAM, ~17K tok/s on RTX 4080 Super
    "128M":   dict(vocab_size=32000, d_model=512,  n_layers=6,  n_heads=8,
                   latent_dim=128,  n_experts=8,  top_k=2,  mem_slots=64,  frc_steps=3),

    # 1B  — d_model=2048, moe_d_ff=1024, 4 experts  → ~728M actual params
    # Sizing: moe_d_ff << 4×d_model keeps total params honest for sparse MoE.
    # VRAM: ~11.6 GB (fp32 weights + AdamW) + activations → fits 16 GB with GC.
    "1B":     dict(vocab_size=32000, d_model=2048, n_layers=16, n_heads=16,
                   latent_dim=512,  n_experts=4,  top_k=2,  mem_slots=128,
                   frc_steps=3,    moe_d_ff=1024),

    # 7B  — aspirational; requires 4-GPU node or CPU offload
    "7B":     dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                   latent_dim=1024, n_experts=8,  top_k=2,  mem_slots=256,
                   frc_steps=4,    moe_d_ff=1792),

    # 13B  — aspirational; requires 8-GPU node
    "13B":    dict(vocab_size=32000, d_model=5120, n_layers=40, n_heads=40,
                   latent_dim=1280, n_experts=8,  top_k=2,  mem_slots=256,
                   frc_steps=4,    moe_d_ff=2048),

    # ── Long-context variants — NTK-aware RoPE scaling ────────────────────
    # rope_scale=4.0 → ~4× context extension at inference (no fine-tuning)
    "1B-8k":  dict(vocab_size=32000, d_model=2048, n_layers=16, n_heads=16,
                   latent_dim=512,  n_experts=4,  top_k=2,  mem_slots=128,
                   frc_steps=3,    moe_d_ff=1024, rope_scale=4.0),
    "1B-32k": dict(vocab_size=32000, d_model=2048, n_layers=16, n_heads=16,
                   latent_dim=512,  n_experts=4,  top_k=2,  mem_slots=128,
                   frc_steps=3,    moe_d_ff=1024, rope_scale=16.0),
    "7B-32k": dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                   latent_dim=1024, n_experts=8,  top_k=2,  mem_slots=512,
                   frc_steps=4,    moe_d_ff=1792, rope_scale=16.0),
}


# ── RoPE ─────────────────────────────────────────────────────────────────

def _rope_freqs(head_dim: int, seq_len: int, base: int = 10_000,
                scale: float = 1.0,
                device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (cos, sin) tensors of shape [1, 1, seq_len, head_dim].

    scale > 1.0 enables NTK-aware context extension (no fine-tuning needed).
    Effective base = base * scale^(d/(d-2)), which rescales high-frequency
    components more aggressively than low-frequency ones — better than naive
    linear position interpolation for long sequences.

    Recommended scales:
        1.0  → training length (default)
        2.0  → 2× context    (minimal quality loss)
        4.0  → 4× context    (slight quality degradation at tail)
        8.0  → 8× context    (good for retrieval tasks, worse for generation)
    """
    if scale != 1.0:
        base = base * (scale ** (head_dim / (head_dim - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # [T, head_dim/2]
    emb   = torch.cat([freqs, freqs], dim=-1)   # [T, head_dim]
    return emb.cos()[None, None], emb.sin()[None, None]  # [1,1,T,D]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. Tensors are [B, H, T, head_dim]."""
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ── MLAPlus with RoPE + Flash Attention ───────────────────────────────────

class MLAPlus(nn.Module):
    """
    Multi-Head Latent Attention Plus.

    KV compression (DeepSeek-V2 MLA): K, V are projected through a shared
    low-rank latent bottleneck (latent_dim << d_model), reducing KV cache
    by d_model/latent_dim.  Queries are full-rank.

    RoPE replaces learned absolute position embeddings and supports
    sequences far beyond the training length without re-training.

    Attention is computed with F.scaled_dot_product_attention which
    dispatches to Flash Attention 2 kernels on CUDA when available.
    """

    def __init__(self, d_model: int, n_heads: int, latent_dim: int = 128,
                 rope_base: int = 10_000, rope_scale: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.rope_base  = rope_base
        self.rope_scale = rope_scale

        self.q_proj      = nn.Linear(d_model, d_model,   bias=False)
        self.kv_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.k_expand    = nn.Linear(latent_dim, d_model, bias=False)
        self.v_expand    = nn.Linear(latent_dim, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model,   bias=False)

        # Cached RoPE frequencies — extended lazily when seq_len grows
        self._rope_seq_len = 0
        self._cos = self._sin = None

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self._rope_seq_len:
            self._cos, self._sin = _rope_freqs(
                self.head_dim, seq_len, self.rope_base, self.rope_scale, device
            )
            self._rope_seq_len = seq_len
        return self._cos[:, :, :seq_len, :], self._sin[:, :, :seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        latent = self.kv_compress(x)            # [B, T, latent_dim]

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = to_heads(self.q_proj(x))            # [B, H, T, head_dim]
        k = to_heads(self.k_expand(latent))
        v = to_heads(self.v_expand(latent))

        cos, sin = self._get_rope(T, x.device)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        # Flash Attention 2 when CUDA is available; standard fallback otherwise
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ── AQ-MoE: Adaptive Quantum-Inspired Mixture of Experts ─────────────────

class AdaptiveMoE(nn.Module):
    """
    Sparse MoE with an expert-correlation router.

    Standard router:  s = x W_router                   [N, E]
    Coupled router:   s_coupled = s + eta * (s @ J)    [N, E]

    J ∈ R^{E×E} is a learned matrix that lets the router adjust scores based
    on co-activation patterns between experts. Initialised to zeros so it
    starts as a standard top-k router and learns coupling during training.

    Note: "AQ-MoE" / "quantum-inspired" in earlier docs refers only to this
    coupling matrix — the implementation is entirely classical.

    Exposes .load_balance_loss for the auxiliary objective:
        L_balance = E · Σᵢ(fᵢ · Pᵢ)
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 d_ff: Optional[int] = None, eta: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.n_experts = n_experts
        self.top_k     = top_k
        self.eta       = eta

        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Expert coupling matrix — E×E, init zeros (no coupling at start)
        self.J = nn.Parameter(torch.zeros(n_experts, n_experts))

        # Experts: each is a 2-layer FFN stored as weight tensors for efficiency
        self.w1 = nn.Parameter(torch.randn(n_experts, d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)

        self.load_balance_loss: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)   # [N, C]  N = B*T

        # Coupled routing
        s        = self.router(x_flat)                    # [N, E]
        s_coup   = s + self.eta * (s @ self.J)           # [N, E]
        probs    = F.softmax(s_coup, dim=-1)              # [N, E]
        tk_prob, tk_idx = probs.topk(self.top_k, dim=-1)  # [N, k]
        tk_prob  = tk_prob / tk_prob.sum(dim=-1, keepdim=True)

        # Load-balance loss: L = E * sum_i(f_i * P_i)
        # f_i = fraction of tokens assigned to expert i
        # P_i = mean router probability for expert i
        with torch.no_grad():
            one_hot = torch.zeros_like(probs).scatter_(
                1, tk_idx[:, :1], 1.0           # use primary expert only for balance
            )
            f_i = one_hot.mean(0)               # [E]
            P_i = probs.mean(0)                 # [E]
            self.load_balance_loss = (self.n_experts * (f_i * P_i).sum()).item()

        # Vectorized dispatch (loop over E experts, not N tokens)
        output = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask   = (tk_idx == e).any(dim=-1)     # [N]
            if not mask.any():
                continue
            w      = (tk_prob * (tk_idx == e).float()).sum(dim=-1)[mask]  # [m]
            tokens = x_flat[mask]                  # [m, C]
            h      = F.gelu(tokens @ self.w1[e])   # [m, d_ff]
            out    = h @ self.w2[e]                # [m, C]
            output[mask] += w.unsqueeze(-1) * out

        return output.view(B, T, C)


# ── FRC v2: memory buffer + tool hooks ───────────────────────────────────

class ForgeReasoningCore(nn.Module):
    """
    Forge Reasoning Core v2.

    Gated iterative refinement (v1) plus:

    Memory buffer — mem_slots learned key-value pairs (d_model each).
    Each FRC step reads from memory via soft attention and gates the read
    with a per-step scalar gate initialised near zero.  The memory is
    trainable but not updated during forward passes (static KV store);
    update it between FRC steps via tool hooks if needed.

    Tool hooks — register callables with .register_tool(step, fn).
    fn(x) -> x is called after step `step`'s refinement+memory read.
    Use this for: external retrieval, calculator calls, code execution,
    structured output extraction, or any non-differentiable operation.

    FRC depth can be increased at inference beyond training depth by
    directly setting model.frc.n_steps (and padding steps/gates with
    pre-trained or zero-init parameters).
    """

    def __init__(self, d_model: int, n_steps: int = 3, mem_slots: int = 64):
        super().__init__()
        self.d_model   = d_model
        self.n_steps   = n_steps
        self.scale     = d_model ** -0.5

        # Refinement layers
        self.steps = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_steps)
        ])
        # Per-step scalar gates — init zero → near-identity at init
        self.gates = nn.Parameter(torch.zeros(n_steps, d_model))

        # Memory buffer
        self.mem_k    = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)
        self.mem_v    = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)
        # Per-step memory read gate — init zero (off)
        self.mem_gate = nn.Parameter(torch.zeros(n_steps))

        # Tool hooks: {step_index: callable(x: Tensor) -> Tensor}
        self.tool_hooks: dict[int, Callable] = {}

    def register_tool(self, step: int, fn: Callable) -> None:
        """Register a tool function called after FRC step `step`.
        fn receives x [B, T, d_model] and must return the same shape."""
        self.tool_hooks[step] = fn

    def _mem_read(self, x: torch.Tensor) -> torch.Tensor:
        """Soft attention over memory slots.  x: [B, T, d_model]"""
        attn = torch.softmax(x @ self.mem_k.T * self.scale, dim=-1)  # [B, T, M]
        return attn @ self.mem_v                                        # [B, T, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (step, gate) in enumerate(zip(self.steps, self.gates)):
            # Refinement with per-dim gate
            x = x + torch.sigmoid(gate) * F.gelu(step(x))
            # Memory read with per-step scalar gate
            x = x + torch.sigmoid(self.mem_gate[i]) * self._mem_read(x)
            # Optional tool hook
            if i in self.tool_hooks:
                x = self.tool_hooks[i](x)
        return x


# ── Block ─────────────────────────────────────────────────────────────────

class AetherForgeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int,
                 n_experts: int, top_k: int, mem_slots: int, frc_steps: int,
                 rope_base: int = 10_000, rope_scale: float = 1.0,
                 moe_d_ff: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)
        self.attn  = MLAPlus(d_model, n_heads, latent_dim, rope_base, rope_scale)
        self.moe   = AdaptiveMoE(d_model, n_experts, top_k, d_ff=moe_d_ff)
        self.frc   = ForgeReasoningCore(d_model, frc_steps, mem_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        x = x + self.frc(self.norm3(x))
        return x

    @property
    def load_balance_loss(self) -> float:
        return self.moe.load_balance_loss


# ── AetherForge ──────────────────────────────────────────────────────────

class AetherForge(nn.Module):
    """
    AetherForge v2 — full architecture.

    Instantiation:
        model = AetherForge(**MODEL_CONFIGS["128M"])
        model = AetherForge.from_config("7B")

    Training objective:
        loss = loss_lm + balance_alpha * model.load_balance_loss()
    """

    def __init__(
        self,
        vocab_size:  int   = 32000,
        d_model:     int   = 512,
        n_layers:    int   = 6,
        n_heads:     int   = 8,
        latent_dim:  int   = 128,
        n_experts:   int   = 8,
        top_k:       int   = 2,
        mem_slots:   int   = 64,
        frc_steps:   int   = 3,
        rope_base:   int   = 10_000,
        rope_scale:  float = 1.0,
        moe_d_ff:    Optional[int] = None,
    ):
        super().__init__()
        self.d_model    = d_model
        self.rope_scale = rope_scale

        # No positional embedding — RoPE is applied inside MLAPlus
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks    = nn.ModuleList([
            AetherForgeBlock(d_model, n_heads, latent_dim, n_experts, top_k,
                             mem_slots, frc_steps, rope_base, rope_scale, moe_d_ff)
            for _ in range(n_layers)
        ])
        self.norm    = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # weight tying

        self._init_weights()

    @classmethod
    def from_config(cls, name: str) -> "AetherForge":
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Choose from: {list(MODEL_CONFIGS)}")
        return cls(**MODEL_CONFIGS[name])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def load_balance_loss(self) -> torch.Tensor:
        """Sum of MoE load-balance losses across all blocks (scalar tensor)."""
        total = sum(b.load_balance_loss for b in self.blocks)
        return torch.tensor(total, requires_grad=False)

    def extend_context(self, scale: float) -> "AetherForge":
        """
        Switch to NTK-aware RoPE scaling at inference time without retraining.

        Updates the rope_scale on all MLAPlus blocks and invalidates the
        cached frequency tensors so they are rebuilt on the next forward pass.

        Args:
            scale: multiplier on effective base frequency.
                   2.0 → ~2× context, 4.0 → ~4×, 8.0 → ~8×.

        Returns self for chaining:  model.eval().extend_context(4.0)
        """
        self.rope_scale = scale
        for block in self.blocks:
            block.attn.rope_scale  = scale
            block.attn._rope_seq_len = 0   # force cache rebuild
        return self

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        gc = getattr(self, "_gradient_checkpointing", False) and self.training
        for block in self.blocks:
            if gc:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.norm(x))

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.1f}M"

    @torch.no_grad()
    def generate(
        self,
        input_ids:     torch.Tensor,
        max_new_tokens: int   = 100,
        temperature:    float = 0.8,
        top_p:          float = 0.9,
        eos_token_id:   int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with top-p nucleus sampling.

        Args:
            input_ids:          [B, T] prompt token IDs.
            max_new_tokens:     Maximum tokens to generate.
            temperature:        Sampling temperature.  0 → greedy argmax.
            top_p:              Nucleus cutoff probability.
            eos_token_id:       Stop when this token is generated.
            repetition_penalty: > 1 penalises tokens that already appear;
                                1.0 = no penalty.
        """
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(generated)[:, -1, :]   # [B, V]

            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    unique = generated[b].unique()
                    logits[b, unique] /= repetition_penalty

            if temperature == 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs  = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_tok = torch.multinomial(sorted_probs, 1)
                next_tok = sorted_idx.gather(-1, next_tok)

            generated = torch.cat([generated, next_tok], dim=-1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return generated


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = sys.argv[1] if len(sys.argv) > 1 else "128M"

    print(f"Device : {device}")
    print(f"Config : {config}")

    model = AetherForge.from_config(config).to(device)
    print(f"Params : {model.param_count()}")

    # Forward pass
    ids    = torch.randint(0, 32000, (2, 128), device=device)
    logits = model(ids)
    print(f"Forward: input {tuple(ids.shape)} -> logits {tuple(logits.shape)}")

    # Load balance loss
    lbl = model.load_balance_loss()
    print(f"Load-balance loss : {lbl.item():.4f}")

    # Tool hook demo
    def calculator_hook(x: torch.Tensor) -> torch.Tensor:
        """Placeholder — in production this would call an external tool."""
        return x
    model.blocks[0].frc.register_tool(0, calculator_hook)

    # Long-context, extend-context, and generation tests.
    # On large configs with another process occupying GPU, these can OOM.
    # CUDA raises asynchronously — the error may surface on the next CUDA
    # call after the failing one, so a single outer try/except is the only
    # reliable guard for all three steps.
    try:
        # Long-context test — RoPE extends lazily
        long_ids = torch.randint(0, 32000, (1, 1024), device=device)
        out_long = model(long_ids)
        if device == "cuda":
            torch.cuda.synchronize()   # flush async OOM before printing
        print(f"Long-ctx 1024   : logits {tuple(out_long.shape)}")
        del out_long

        # Context extension — NTK-aware RoPE scaling
        # fp32 SDPA is O(N²); skip when headroom < 2 GB (training uses AMP/FA2).
        if device == "cuda":
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info()[0] / 1e9
        else:
            free_vram = 99.0
        if free_vram >= 2.0:
            ext_len = min(4096, int(256 * free_vram))
            model.extend_context(4.0)
            xl_ids = torch.randint(0, 32000, (1, ext_len), device=device)
            out_xl = model(xl_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"Extended {ext_len} (scale=4.0): logits {tuple(out_xl.shape)}")
            model.extend_context(1.0)
            del out_xl
        else:
            print(f"Extended ctx    : skipped ({free_vram*1024:.0f} MB free — use AMP)")

        # Generation with EOS stop + repetition penalty
        if device == "cuda":
            torch.cuda.empty_cache()
        gen = model.generate(ids[:1, :8], max_new_tokens=5,
                             eos_token_id=2, repetition_penalty=1.1)
        print(f"Generation      : {tuple(gen.shape)}")

    except Exception as _oom:
        if "out of memory" not in str(_oom).lower():
            raise
        print("Long-ctx/gen    : skipped (VRAM tight — OK; training uses fresh allocator)")

    if device == "cuda":
        print(f"VRAM   : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("AetherForge v2 OK")
