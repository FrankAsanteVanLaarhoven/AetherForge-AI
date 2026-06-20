"""
Save the 4-bit quantized model to disk so future loads skip re-quantization.
Output: ~/.cache/huggingface/hub/llama3-8b-4bit-nf4/

Usage:
    conda run -n ml-torch python save_quantized.py

After saving, load it with:
    model = AutoModelForCausalLM.from_pretrained(
        "~/.cache/huggingface/hub/llama3-8b-4bit-nf4",
        device_map="auto",
    )
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
SAVE_DIR = os.path.expanduser("~/.cache/huggingface/hub/llama3-8b-4bit-nf4")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading {MODEL_ID} with 4-bit NF4 quantization ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)

print(f"Saving quantized model to {SAVE_DIR} ...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Done.")
print(f"VRAM at save time: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
