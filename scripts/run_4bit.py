"""
Load Llama-3-8B Base in 4-bit NF4 (bitsandbytes) and run inference.

Memory: ~5.5GB VRAM (fits comfortably in RTX 4080 16GB).

Usage:
    conda run -n ml-torch python run_4bit.py
    conda run -n ml-torch python run_4bit.py --prompt "The robot navigated the corridor"
    conda run -n ml-torch python run_4bit.py --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — best quality for LLMs
    bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4 bits/param
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit NF4 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM used: {vram:.1f} GB")
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The hospital corridor robot detected an obstacle and")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.interactive:
        print("\nInteractive mode. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                print(generate(tokenizer, model, prompt, args.max_tokens))
                print()
            except KeyboardInterrupt:
                break
    else:
        print(f"\nPrompt: {args.prompt}\n")
        print(generate(tokenizer, model, args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
