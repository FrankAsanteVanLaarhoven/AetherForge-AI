"""
Evaluate a fine-tuned Qwen2.5-VL model on text and vision tasks.

Usage:
    # Text only (base model)
    conda run -n ml-torch python scripts/evaluate_model.py

    # Text + vision on a single image
    conda run -n ml-torch python scripts/evaluate_model.py \
        --image multimodal_example/images/00_0_hospital_corridor.png

    # Full suite with LoRA adapter
    conda run -n ml-torch python scripts/evaluate_model.py \
        --lora-path ./outputs/qwen25_vl_lora/final \
        --benchmark all \
        --image-dir multimodal_example/images

    # Before/after comparison (base vs. fine-tuned)
    conda run -n ml-torch python scripts/evaluate_model.py \
        --lora-path ./outputs/qwen25_vl_lora/final \
        --benchmark all \
        --image-dir multimodal_example/images \
        --compare-base
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

TEXT_BENCH = [
    {"instruction": "What is 17 multiplied by 23?",
     "expected_contains": ["391"]},
    {"instruction": "Write a Python function that returns the factorial of n.",
     "expected_contains": ["def", "factorial", "return"]},
    {"instruction": "Explain self-attention in one sentence.",
     "expected_contains": ["attention", "query", "key"]},
    {"instruction": "What is the capital of France?",
     "expected_contains": ["Paris"]},
    {"instruction": "Name three layers in the Transformer architecture.",
     "expected_contains": ["attention", "feed", "norm"]},
    {"instruction": "What does MoE stand for in machine learning?",
     "expected_contains": ["mixture", "experts"]},
    {"instruction": "Solve: 2x + 5 = 17. What is x?",
     "expected_contains": ["6"]},
    {"instruction": "What is gradient checkpointing used for?",
     "expected_contains": ["memory", "vram", "activation"]},
    {"instruction": "What is LoRA fine-tuning?",
     "expected_contains": ["low-rank", "rank", "adapter", "parameter"]},
    {"instruction": "Name two Chinese large language models released in 2024-2025.",
     "expected_contains": ["qwen", "deepseek", "kimi", "glm", "baichuan"]},
]

# Hospital-scene questions matched to dummy dataset labels
VISION_QUESTIONS = [
    {
        "instruction": "What type of environment or zone is depicted in this image?",
        "expected_contains": [
            "hospital", "corridor", "room", "medical", "pharmacy", "cafeteria",
            "elevator", "reception", "radiology", "supply", "exit", "operating",
        ],
    },
    {
        "instruction": "What text or label can you read in this image?",
        "expected_contains": [
            "corridor", "room", "pharmacy", "reception", "exit", "cafeteria",
            "elevator", "radiology", "supply", "operating",
        ],
    },
    {
        "instruction": "What colour is the background of this image?",
        "expected_contains": [
            "blue", "green", "white", "gray", "grey", "pink", "red",
            "yellow", "purple", "light", "dark",
        ],
    },
]


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _ngram_overlap(candidate: str, reference: str, n: int = 2) -> float:
    """Simple n-gram precision — lightweight BLEU proxy."""
    def ngrams(text, n):
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    cand_ng = ngrams(candidate, n)
    ref_ng  = ngrams(reference, n)
    if not cand_ng:
        return 0.0
    matches = sum(min(cand_ng[k], ref_ng[k]) for k in cand_ng)
    return matches / sum(cand_ng.values())


def _keyword_hit(response: str, keywords: list[str]) -> bool:
    lower = response.lower()
    return any(kw.lower() in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(lora_path: str | None = None, label: str = "model"):
    print(f"\nLoading {MODEL_ID} in 4-bit NF4 [{label}] ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        print(f"  Loading LoRA weights from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Ready. VRAM: {vram:.1f} GB")
    return model, processor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, processor, prompt: str, image_path: str | None = None,
             max_new_tokens: int = 200) -> str:
    if image_path:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=text, images=[image],
                           return_tensors="pt").to(model.device)
    else:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_ids, skip_special_tokens=True)


def compute_perplexity(model, processor, texts: list[str],
                       max_length: int = 128) -> float:
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = processor(text=text, return_tensors="pt",
                        max_length=max_length, truncation=True).to(model.device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
        total_loss   += out.loss.item() * (input_ids.shape[-1] - 1)
        total_tokens += input_ids.shape[-1] - 1
    return math.exp(total_loss / total_tokens) if total_tokens else float("inf")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_text_benchmark(model, processor, label: str = "") -> tuple[float, list]:
    tag = f" [{label}]" if label else ""
    print("\n" + "=" * 65)
    print(f"TEXT BENCHMARK{tag}")
    print("=" * 65)
    passed, results = 0, []
    for item in TEXT_BENCH:
        t0 = time.time()
        response = generate(model, processor, item["instruction"])
        elapsed  = time.time() - t0
        hit      = _keyword_hit(response, item["expected_contains"])
        passed  += hit
        results.append({"q": item["instruction"], "a": response, "pass": hit})
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {item['instruction'][:57]:<57} ({elapsed:.1f}s)")
        if not hit:
            print(f"       Expected one of: {item['expected_contains']}")
            print(f"       Got: {response[:120]}")
    score = passed / len(TEXT_BENCH)
    print(f"\nText score: {passed}/{len(TEXT_BENCH)} ({score*100:.0f}%)")
    return score, results


def run_vision_benchmark(model, processor,
                         image_source: str,
                         label: str = "") -> tuple[float, list]:
    """
    image_source: path to a single image file, or a directory of images.
    Runs VISION_QUESTIONS against each image and reports keyword accuracy.
    """
    tag = f" [{label}]" if label else ""
    print("\n" + "=" * 65)
    print(f"VISION BENCHMARK{tag}")
    print("=" * 65)

    src = Path(image_source)
    if src.is_dir():
        images = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")) + \
                 sorted(src.glob("*.jpeg"))
        images = images[:10]   # cap at 10 to keep it fast
    else:
        images = [src]

    if not images:
        print(f"No images found at {image_source}")
        return 0.0, []

    total_q   = len(VISION_QUESTIONS) * len(images)
    passed    = 0
    results   = []
    bleu_sum  = 0.0

    for img_path in images:
        img_label = img_path.stem.replace("_", " ")
        print(f"\n  Image: {img_path.name}")
        for q in VISION_QUESTIONS:
            t0       = time.time()
            response = generate(model, processor, q["instruction"],
                                str(img_path))
            elapsed  = time.time() - t0
            hit      = _keyword_hit(response, q["expected_contains"])
            # Bigram overlap against the expected keywords joined as a phrase
            ref_phrase = " ".join(q["expected_contains"])
            bleu       = _ngram_overlap(response, ref_phrase, n=1)
            bleu_sum  += bleu
            passed    += hit
            status     = "PASS" if hit else "FAIL"
            print(f"    [{status}] {q['instruction'][:50]:<50} ({elapsed:.1f}s)")
            if not hit:
                print(f"           Got: {response[:100]}")
            results.append({
                "image": str(img_path),
                "q": q["instruction"],
                "a": response,
                "pass": hit,
                "unigram_overlap": round(bleu, 3),
            })

    score      = passed / total_q if total_q else 0.0
    avg_bleu   = bleu_sum / total_q if total_q else 0.0
    print(f"\nVision score: {passed}/{total_q} ({score*100:.0f}%)")
    print(f"Avg unigram overlap: {avg_bleu:.3f}")
    return score, results


def run_perplexity(model, processor) -> float:
    print("\n" + "=" * 65)
    print("PERPLEXITY")
    print("=" * 65)
    texts = [
        "The Transformer architecture uses self-attention to model relationships.",
        "Mixture of Experts routes each token to a subset of specialist networks.",
        "Gradient checkpointing reduces activation memory at the cost of compute.",
        "Low-rank adaptation fine-tunes only a small fraction of model parameters.",
        "Qwen2.5-VL is a multimodal vision-language model from Alibaba DAMO.",
    ]
    ppl = compute_perplexity(model, processor, texts)
    print(f"Perplexity: {ppl:.2f}  (lower = better)")
    return ppl


# ---------------------------------------------------------------------------
# Before/after comparison
# ---------------------------------------------------------------------------

def compare_models(lora_path: str, image_source: str | None,
                   benchmark: str, output_path: str):
    """Run benchmark on base then on LoRA-tuned model; print diff table."""
    base_model, base_proc = load_model(label="base")
    base_res = _collect_results(base_model, base_proc, benchmark, image_source)
    del base_model   # free VRAM before loading LoRA
    torch.cuda.empty_cache()

    lora_model, lora_proc = load_model(lora_path, label="fine-tuned")
    lora_res = _collect_results(lora_model, lora_proc, benchmark, image_source)

    _print_comparison(base_res, lora_res)

    combined = {"base": base_res, "finetuned": lora_res}
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nComparison saved to {output_path}")


def _collect_results(model, processor, benchmark: str,
                     image_source: str | None) -> dict:
    res = {}
    if benchmark in ("text", "all"):
        score, details = run_text_benchmark(model, processor)
        res["text"] = {"score": score, "details": details}
    if benchmark in ("vision", "all"):
        if image_source:
            score, details = run_vision_benchmark(model, processor, image_source)
            res["vision"] = {"score": score, "details": details}
    if benchmark in ("perplexity", "all"):
        res["perplexity"] = run_perplexity(model, processor)
    return res


def _print_comparison(base: dict, fine: dict):
    print("\n" + "=" * 65)
    print("BASE vs. FINE-TUNED COMPARISON")
    print("=" * 65)
    metrics = []
    if "text" in base and "text" in fine:
        metrics.append(("Text accuracy",
                         f"{base['text']['score']*100:.0f}%",
                         f"{fine['text']['score']*100:.0f}%"))
    if "vision" in base and "vision" in fine:
        metrics.append(("Vision accuracy",
                         f"{base['vision']['score']*100:.0f}%",
                         f"{fine['vision']['score']*100:.0f}%"))
    if "perplexity" in base and "perplexity" in fine:
        metrics.append(("Perplexity",
                         f"{base['perplexity']:.2f}",
                         f"{fine['perplexity']:.2f}"))
    print(f"  {'Metric':<22} {'Base':>12} {'Fine-tuned':>12}  {'Delta':>8}")
    print("  " + "-" * 58)
    for name, b_val, f_val in metrics:
        try:
            b_num = float(b_val.strip("%"))
            f_num = float(f_val.strip("%"))
            delta = f"{f_num - b_num:+.1f}{'%' if '%' in b_val else ''}"
        except ValueError:
            delta = "n/a"
        print(f"  {name:<22} {b_val:>12} {f_val:>12}  {delta:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lora-path", default=None,
                        help="LoRA adapter directory to load on top of base model")
    parser.add_argument("--image", default=None,
                        help="Single image file for vision benchmark")
    parser.add_argument("--image-dir",
                        default="multimodal_example/images",
                        help="Directory of images — up to 10 are tested")
    parser.add_argument("--benchmark",
                        choices=["text", "vision", "perplexity", "all"],
                        default="text")
    parser.add_argument("--compare-base", action="store_true",
                        help="Run on base model first, then on LoRA, print diff table")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    image_source = args.image or (
        args.image_dir if args.benchmark in ("vision", "all") else None
    )

    if args.compare_base:
        if not args.lora_path:
            parser.error("--compare-base requires --lora-path")
        compare_models(args.lora_path, image_source, args.benchmark, args.output)
        return

    model, processor = load_model(args.lora_path)
    all_results = {}

    if args.benchmark in ("text", "all"):
        score, details = run_text_benchmark(model, processor)
        all_results["text"] = {"score": score, "details": details}

    if args.benchmark in ("vision", "all"):
        if not image_source:
            print("\nVision benchmark skipped — pass --image or --image-dir")
        else:
            score, details = run_vision_benchmark(model, processor, image_source)
            all_results["vision"] = {"score": score, "details": details}

    if args.benchmark in ("perplexity", "all"):
        ppl = run_perplexity(model, processor)
        all_results["perplexity"] = ppl

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    if "text" in all_results:
        print(f"  Text:         {all_results['text']['score']*100:.0f}%")
    if "vision" in all_results:
        print(f"  Vision:       {all_results['vision']['score']*100:.0f}%")
    if "perplexity" in all_results:
        print(f"  Perplexity:   {all_results['perplexity']:.2f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
