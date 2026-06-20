"""
Generate AetherForge architecture diagram — docs/architecture.png

Run:
    conda run -n ml-torch python scripts/generate_architecture_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

BG      = "#0D1117"
SURFACE = "#161B22"
BORDER  = "#30363D"
GOLD    = "#E3B341"
BLUE    = "#58A6FF"
GREEN   = "#3FB950"
PURPLE  = "#BC8CFF"
TEXT_P  = "#E6EDF3"
TEXT_S  = "#8B949E"


def box(ax, x, y, w, h, face, lines, colors=None, fontsize=7.2, radius=0.012):
    """Draw a rounded box; lines = list of (text, color) or just list of str."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=BORDER, linewidth=1.1, zorder=3,
    )
    ax.add_patch(patch)
    n = len(lines)
    for i, line in enumerate(lines):
        txt   = line if isinstance(line, str) else line[0]
        color = (TEXT_P if isinstance(line, str) else line[1])
        frac  = (i + 0.5) / n
        ax.text(x + w / 2, y + h * (1 - frac), txt,
                ha="center", va="center",
                fontsize=fontsize, color=color, zorder=4,
                fontweight=("bold" if i == 0 else "normal"))


def arr(ax, x1, y1, x2, y2, color=TEXT_S, style="->"):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=1.3, mutation_scale=10),
                zorder=2)


fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────
ax.text(0.5, 0.972, "AetherForge v2  —  Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color=GOLD)
ax.text(0.5, 0.950,
        "128M → 13B  ·  RoPE  ·  Flash Attention 2  ·  AQ-MoE Coupling Matrix  ·  FRC Memory + Tool Hooks",
        ha="center", va="center", fontsize=8.5, color=TEXT_S)

# ── Outer block dashed border ─────────────────────────────────────────────
outer = FancyBboxPatch(
    (0.18, 0.08), 0.62, 0.84,
    boxstyle="round,pad=0,rounding_size=0.015",
    facecolor="none", edgecolor=BLUE, linewidth=1.0,
    linestyle="dashed", zorder=1,
)
ax.add_patch(outer)
ax.text(0.185, 0.930, "× N  AetherForge Blocks",
        ha="left", va="center", fontsize=8, color=BLUE, fontstyle="italic")

# ── Layout constants ──────────────────────────────────────────────────────
BH = 0.062          # small box height
TH = 0.155          # tall component box height
BW = 0.175          # box width

# Column x-centres for: input, MLA+, AQ-MoE, FRC, output
CX = [0.06, 0.295, 0.495, 0.695, 0.06]

# ── Input / Embedding ─────────────────────────────────────────────────────
box(ax, CX[0]-0.05, 0.855, 0.14, BH, "#1C2128",
    [("Token IDs", TEXT_P), ("[B, T]  →  Embedding", TEXT_S)])
arr(ax, CX[0], 0.855, CX[0], 0.820)

# ── RMSNorm → MLA+ ────────────────────────────────────────────────────────
# Norm
box(ax, CX[1]-BW/2, 0.840, BW, BH, "#1a2332",
    [("RMSNorm", TEXT_P), ("pre-norm", TEXT_S)])
arr(ax, CX[1], 0.840, CX[1], 0.800)

# MLA+ component
mla_y = 0.620
box(ax, CX[1]-BW/2, mla_y, BW, TH, "#152236", [
    ("MLAPlus", BLUE),
    ("KV compress: x → latent_dim", TEXT_S),
    ("K,  Q  ←  RoPE(pos,  θ)", TEXT_P),
    ("Flash Attention 2  (is_causal=True)", TEXT_P),
    ("KV cache  ×  4  smaller", GREEN),
])
arr(ax, CX[1], 0.800, CX[1], mla_y + TH)
arr(ax, CX[1], mla_y, CX[1], mla_y - 0.030)

# ── RMSNorm → AQ-MoE ─────────────────────────────────────────────────────
box(ax, CX[2]-BW/2, 0.840, BW, BH, "#1a2332",
    [("RMSNorm", TEXT_P), ("pre-norm", TEXT_S)])
arr(ax, CX[2], 0.840, CX[2], 0.800)

moe_y = 0.620
box(ax, CX[2]-BW/2, moe_y, BW, TH, "#2e1e00", [
    ("AQ-MoE", GOLD),
    ("s  =  x W_router", TEXT_S),
    ("s'  =  s  +  η (s @ J)", GOLD),
    ("J ∈ ℝᴱˣᴱ  (learned, init 0)", TEXT_P),
    ("Top-K dispatch  +  Load balance", GREEN),
])
arr(ax, CX[2], 0.800, CX[2], moe_y + TH)
arr(ax, CX[2], moe_y, CX[2], moe_y - 0.030)

# ── RMSNorm → FRC ────────────────────────────────────────────────────────
box(ax, CX[3]-BW/2, 0.840, BW, BH, "#1a2332",
    [("RMSNorm", TEXT_P), ("pre-norm", TEXT_S)])
arr(ax, CX[3], 0.840, CX[3], 0.800)

frc_y = 0.620
box(ax, CX[3]-BW/2, frc_y, BW, TH, "#22183a", [
    ("Forge Reasoning Core", PURPLE),
    ("h  ←  h  +  σ(g) · GELU(Wh)", TEXT_S),
    ("h  ←  h  +  σ(γ) · MemRead(h)", PURPLE),
    ("Memory:  K,V ∈ ℝᴹˣᴰ  (64 slots)", TEXT_P),
    ("register_tool(step,  fn)", GREEN),
])
arr(ax, CX[3], 0.800, CX[3], frc_y + TH)
arr(ax, CX[3], frc_y, CX[3], frc_y - 0.030)

# ── Horizontal flow inside block ──────────────────────────────────────────
flow_y = 0.700
arr(ax, CX[0]+0.04, 0.820, CX[1]-BW/2, flow_y + 0.030, color=TEXT_S)
arr(ax, CX[1]+BW/2, flow_y, CX[2]-BW/2, flow_y, color=TEXT_S)
arr(ax, CX[2]+BW/2, flow_y, CX[3]-BW/2, flow_y, color=TEXT_S)

# ── Output path ───────────────────────────────────────────────────────────
# From FRC to output column
out_top_y = 0.530
arr(ax, CX[3], frc_y, CX[3], out_top_y)
ax.annotate("",
            xy=(CX[0]+0.04, out_top_y),
            xytext=(CX[3], out_top_y),
            arrowprops=dict(arrowstyle="-|>", color=TEXT_S,
                            lw=1.3, mutation_scale=10,
                            connectionstyle="arc3,rad=0.0"), zorder=2)

box(ax, CX[0]-0.05, 0.430, 0.14, BH, "#1a2332",
    [("RMSNorm", TEXT_P), ("final", TEXT_S)])
box(ax, CX[0]-0.05, 0.350, 0.14, BH, "#1a2e1a",
    [("LM Head", GREEN), ("weight-tied embed", TEXT_S)])
box(ax, CX[0]-0.05, 0.270, 0.14, BH, "#1a2e1a",
    [("Logits", GREEN), ("[B, T, vocab_size]", TEXT_S)])
arr(ax, CX[0], 0.430, CX[0], 0.430 - 0.002)
arr(ax, CX[0], 0.350, CX[0], 0.350 - 0.002)

# ── Right panel: RoPE callout ─────────────────────────────────────────────
rx = 0.870
box(ax, rx, 0.855, 0.115, 0.062, "#162133", [
    ("RoPE", BLUE),
    ("Lazy freq cache — 1M+ ctx", TEXT_S),
])
# dashed arrow from RoPE to MLA+
ax.annotate("",
            xy=(CX[1]+BW/2, mla_y + TH + 0.010),
            xytext=(rx, 0.882),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0,
                            linestyle="dashed", mutation_scale=8,
                            connectionstyle="arc3,rad=-0.25"), zorder=2)

# ── Right panel: size configs ─────────────────────────────────────────────
ax.text(rx + 0.058, 0.800, "Size Configs",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=GOLD)
configs = [
    ("128M",  "d=512   L=6    H=8    E=8"),
    ("1B",    "d=2048  L=24  H=16  E=16"),
    ("7B",    "d=4096  L=32  H=32  E=64"),
    ("13B",   "d=5120  L=40  H=40  E=64"),
]
for i, (name, spec) in enumerate(configs):
    cy = 0.760 - i * 0.048
    nc = GREEN if name == "128M" else TEXT_P
    box(ax, rx, cy - 0.026, 0.115, 0.036, "#21262D", [
        (f"{name}   {spec}", nc),
    ], fontsize=6.3)

# ── Right panel: training objective ──────────────────────────────────────
box(ax, rx, 0.310, 0.115, 0.100, "#201a00", [
    ("Training Objective", GOLD),
    ("ℒ = ℒ_LM  +  α ℒ_balance", TEXT_P),
    ("ℒ_balance = E Σᵢ fᵢ · Pᵢ", TEXT_S),
    ("--balance-alpha  0.01", GREEN),
], fontsize=6.8)

# ── Right panel: verified results ─────────────────────────────────────────
box(ax, rx, 0.175, 0.115, 0.115, "#0d1f0d", [
    ("Verified (RTX 4080 Super)", GREEN),
    ("Text bench:   10/10  100%", TEXT_P),
    ("Vision bench: 30/30  100%", TEXT_P),
    ("Throughput:  ~2500 tok/s", TEXT_S),
    ("VRAM (128M):   1.45 GB", TEXT_S),
], fontsize=6.3)

# ── Legend ────────────────────────────────────────────────────────────────
legend = [
    (BLUE,   "MLAPlus  —  RoPE + Flash Attention 2 + KV compression"),
    (GOLD,   "AQ-MoE  —  coupled routing via J matrix + load-balance loss"),
    (PURPLE, "FRC  —  memory buffer (64 slots) + tool hooks"),
    (GREEN,  "Verified / active feature"),
]
lx, ly = 0.04, 0.090
for i, (c, lbl) in enumerate(legend):
    ax.plot(lx + 0.008, ly - i * 0.024, "s", color=c, markersize=8, zorder=5)
    ax.text(lx + 0.022, ly - i * 0.024, lbl,
            va="center", fontsize=6.8, color=TEXT_S)

# ── Footer ────────────────────────────────────────────────────────────────
ax.text(0.5, 0.018,
        "AetherForge-AI  ·  github.com/FrankAsanteVanLaarhoven/AetherForge-AI"
        "  ·  Newcastle University  ·  2026",
        ha="center", va="center", fontsize=6.5, color=TEXT_S)

out = Path("docs/architecture.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.15)
plt.close()
print(f"Saved {out}  ({out.stat().st_size // 1024} KB)")
