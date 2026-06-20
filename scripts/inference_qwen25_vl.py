"""
Run inference with Qwen2.5-VL-7B-Instruct (4-bit) or a LoRA fine-tuned version.

Usage:
    # Base model, text
    conda run -n ml-torch python scripts/inference_qwen25_vl.py \
        --prompt "Describe the Transformer architecture"

    # Base model, image + text
    conda run -n ml-torch python scripts/inference_qwen25_vl.py \
        --prompt "What do you see?" --image /path/to/image.jpg

    # Fine-tuned LoRA
    conda run -n ml-torch python scripts/inference_qwen25_vl.py \
        --lora-path ./outputs/qwen25_vl_lora/final \
        --prompt "Explain sparse MoE"

    # Interactive REPL
    conda run -n ml-torch python scripts/inference_qwen25_vl.py --interactive
"""

import argparse
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def load(lora_path=None):
    print(f"Loading {MODEL_ID} in 4-bit NF4 ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Ready. VRAM: {vram:.1f}GB")
    return model, processor


def generate(model, processor, prompt: str, image_path=None,
             max_new_tokens: int = 300, temperature: float = 0.7):
    if image_path:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    else:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new = out_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What can you do?")
    parser.add_argument("--image", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, processor = load(args.lora_path)

    if args.interactive:
        print("\nInteractive mode. Ctrl+C to exit. Prefix image path with 'img:<path> '")
        while True:
            try:
                raw = input("\nYou> ").strip()
                if not raw:
                    continue
                img = None
                if raw.startswith("img:"):
                    parts = raw.split(" ", 1)
                    img = parts[0][4:]
                    raw = parts[1] if len(parts) > 1 else ""
                print(generate(model, processor, raw, img, args.max_tokens, args.temperature))
            except KeyboardInterrupt:
                break
    else:
        print(f"\nPrompt: {args.prompt}")
        if args.image:
            print(f"Image:  {args.image}")
        print("\n" + generate(model, processor, args.prompt, args.image,
                              args.max_tokens, args.temperature))


if __name__ == "__main__":
    main()
