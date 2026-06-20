"""merge_lora.py — Merge a LoRA adapter into its base model weights.

Reads the base model path from adapter_config.json automatically so it
works for any LoRA checkpoint regardless of which base model was used.

Usage:
    conda run -n ml-torch python scripts/merge_lora.py \\
        --lora-path outputs/qwen15b_memory_300steps/final \\
        --output-dir outputs/qwen15b_merged_base \\
        --dtype bfloat16

Load the merged model (no PEFT required):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained("outputs/qwen15b_merged_base")
    model = AutoModelForCausalLM.from_pretrained("outputs/qwen15b_merged_base")
"""

import argparse
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def _read_base_model(lora_path: str) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    cfg_path = Path(lora_path) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found at {lora_path}. "
            "Is this a valid PEFT LoRA directory?"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", "")
    if not base:
        raise ValueError(
            "adapter_config.json has no base_model_name_or_path. "
            "Cannot determine which base model to merge into."
        )
    return base


def merge(lora_path: str, output_dir: str, dtype_str: str = "bfloat16"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"--dtype must be one of {list(_DTYPE_MAP)}; got {dtype_str!r}")
    dtype = _DTYPE_MAP[dtype_str]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_model_id = _read_base_model(lora_path)
    print(f"Base model   : {base_model_id}")
    print(f"LoRA adapter : {lora_path}")
    print(f"Output dir   : {out}")
    print(f"Dtype        : {dtype_str}")
    print()

    print(f"[1/4] Loading base model {base_model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"      Base loaded. VRAM: {vram:.1f} GB")

    print(f"[2/4] Applying LoRA adapter …")
    model = PeftModel.from_pretrained(model, lora_path)

    print(f"[3/4] Merging and unloading LoRA weights …")
    model = model.merge_and_unload()
    print(f"      Merge complete — PEFT removed from model.")

    print(f"[4/4] Saving merged model …")
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(out)

    print()
    print(f"Done. Merged model saved to: {out}")
    print("Load with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  tok   = AutoTokenizer.from_pretrained('{out}')")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{out}', device_map='auto')")
    print()
    print("This is now a plain HF model — no PEFT required. Train a fresh LoRA on")
    print("top of this to avoid LoRA-on-LoRA stacking.")


def main():
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lora-path", required=True,
                   help="Path to the PEFT LoRA adapter directory")
    p.add_argument("--output-dir", default="outputs/merged_model",
                   help="Where to save the merged standalone model")
    p.add_argument("--dtype", default="bfloat16",
                   choices=list(_DTYPE_MAP),
                   help="Model dtype for loading and saving")
    args = p.parse_args()

    merge(args.lora_path, args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
