__version__ = "0.1.0"

from .model import (
    AetherForge,
    MLAPlus,
    AdaptiveMoE,
    ForgeReasoningCore,
    MODEL_CONFIGS,
)

# Backwards-compat alias
SparseMoE = AdaptiveMoE

__all__ = [
    "AetherForge",
    "MLAPlus",
    "AdaptiveMoE",
    "SparseMoE",
    "ForgeReasoningCore",
    "MODEL_CONFIGS",
]
