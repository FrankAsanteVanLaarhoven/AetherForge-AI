"""
scripts/eval_checkpoints.py — evaluate all AetherForge checkpoints.

Metrics:
  1. Held-out perplexity on FineWeb text (char-level models: 128M base, 1B)
  2. Held-out instruction CE on Alpaca test split (distilled model, Qwen tok)
  3. Side-by-side generation quality for 5 prompts (all three models)
  4. Summary table + JSON report

Usage:
    conda run -n ml-torch python scripts/eval_checkpoints.py
    conda run -n ml-torch python scripts/eval_checkpoints.py --output outputs/eval_results
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge, MODEL_CONFIGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_charmodel(checkpoint: str, config: str) -> AetherForge:
    cfg = dict(MODEL_CONFIGS[config])
    cfg["vocab_size"] = 32000
    model = AetherForge(**cfg).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_distill_model(checkpoint: str, config_path: str) -> tuple:
    with open(config_path) as f:
        cfg = json.load(f)
    model = AetherForge(**cfg).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=False
    )
    return model, tokenizer


def charlevel_encode(text: str, vocab_size: int = 32000) -> list[int]:
    return [ord(c) % vocab_size for c in text]


def perplexity_on_chunks(
    model: AetherForge,
    chunks: list[str],
    vocab_size: int = 32000,
    seq_len: int = 256,
) -> dict:
    """Compute mean CE loss and perplexity over a list of text chunks."""
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for text in chunks:
            ids = charlevel_encode(text[:seq_len + 1], vocab_size)
            if len(ids) < 4:
                continue
            ids_t = torch.tensor([ids[:seq_len + 1]], device=DEVICE, dtype=torch.long)
            input_ids = ids_t[:, :-1]
            labels    = ids_t[:, 1:]

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            total_loss   += loss.item() * labels.numel()
            total_tokens += labels.numel()

    mean_loss = total_loss / max(total_tokens, 1)
    elapsed   = time.time() - t0
    return {
        "mean_ce_nats":  round(mean_loss, 4),
        "perplexity":    round(math.exp(mean_loss), 2),
        "n_chunks":      len(chunks),
        "n_tokens":      total_tokens,
        "eval_seconds":  round(elapsed, 1),
    }


def perplexity_on_instructions(
    model: AetherForge,
    tokenizer,
    examples: list[dict],
    max_length: int = 256,
) -> dict:
    """Compute response CE (labels on response tokens only) for distilled model."""
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for ex in examples:
            instr = ex.get("instruction", "")
            resp  = ex.get("response", ex.get("output", ""))
            if not resp:
                continue

            text   = f"{instr}\n\n{resp}"
            ids    = tokenizer.encode(text, add_special_tokens=True,
                                      max_length=max_length + 1, truncation=True)
            if len(ids) < 4:
                continue
            # Build labels: -100 for prompt, real tokens for response
            prompt_ids = tokenizer.encode(instr + "\n\n", add_special_tokens=True)
            prompt_len = min(len(prompt_ids), len(ids) - 1)
            labels = [-100] * prompt_len + ids[prompt_len:]
            ids    = ids[:max_length]
            labels = labels[:max_length]

            ids_t = torch.tensor([ids], device=DEVICE, dtype=torch.long)
            lab_t = torch.tensor([labels], device=DEVICE, dtype=torch.long)

            logits = model(ids_t)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), lab_t.view(-1), ignore_index=-100
            )
            n = (lab_t != -100).sum().item()
            if n > 0:
                total_loss   += loss.item() * n
                total_tokens += n

    mean_loss = total_loss / max(total_tokens, 1)
    elapsed   = time.time() - t0
    return {
        "mean_ce_nats":  round(mean_loss, 4),
        "perplexity":    round(math.exp(mean_loss), 2),
        "n_examples":    len(examples),
        "n_tokens":      total_tokens,
        "eval_seconds":  round(elapsed, 1),
    }


def generate_completions(
    model: AetherForge,
    prompts: list[str],
    tokenizer=None,
    vocab_size: int = 32000,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """Generate completions for each prompt."""
    results = []
    eos = tokenizer.eos_token_id if tokenizer else None

    def encode(text):
        if tokenizer:
            return tokenizer.encode(text, add_special_tokens=False)
        return charlevel_encode(text, vocab_size)

    def decode(ids):
        if tokenizer:
            return tokenizer.decode(ids, skip_special_tokens=True)
        return "".join(chr(min(i, 127)) for i in ids)

    with torch.no_grad():
        for prompt in prompts:
            enc = encode(prompt)
            ids = torch.tensor([enc], device=DEVICE, dtype=torch.long)
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                eos_token_id=eos, repetition_penalty=1.1,
            )
            new_ids = out[0, len(enc):].tolist()
            results.append(decode(new_ids).strip())
    return results


# ---------------------------------------------------------------------------
# Held-out data
# ---------------------------------------------------------------------------

def get_fineweb_chunks(n: int = 500, skip: int = 15000, seq_len: int = 256) -> list[str]:
    """Stream n text chunks from FineWeb, skipping the first `skip` items."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True, trust_remote_code=False,
        )
        chunks = []
        for i, item in enumerate(ds):
            if i < skip:
                continue
            text = item.get("text", "")
            if len(text) > seq_len + 1:
                chunks.append(text)
            if len(chunks) >= n:
                break
        print(f"Loaded {len(chunks)} held-out FineWeb chunks (skipped {skip})")
        return chunks
    except Exception as e:
        print(f"FineWeb load failed ({e}), using fallback text")
        sample = "The quick brown fox jumps over the lazy dog. " * 20
        return [sample] * n


def get_alpaca_test(n: int = 200, skip: int = 3000) -> list[dict]:
    """Load n held-out Alpaca examples (not used in distillation training)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=False)
        rows = list(ds)
        test  = []
        for row in rows[skip:skip + n]:
            inp  = row.get("input", "").strip()
            instr = row["instruction"].strip()
            if inp:
                instr = f"{instr}\n\n{inp}"
            resp = row.get("output", "").strip()
            if instr and resp:
                test.append({"instruction": instr, "response": resp})
        print(f"Loaded {len(test)} held-out Alpaca examples (skipped {skip})")
        return test
    except Exception as e:
        print(f"Alpaca load failed ({e})")
        return []


# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

CHAT_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of gradient descent in one paragraph.",
    "Write a Python function to compute the nth Fibonacci number.",
    "What are the main differences between supervised and unsupervised learning?",
    "The history of artificial intelligence begins",
]

INSTRUCTION_PROMPTS = [
    "What is the capital of France?\n\n",
    "Explain the concept of gradient descent in one paragraph.\n\n",
    "Write a Python function to compute the nth Fibonacci number.\n\n",
    "What are the main differences between supervised and unsupervised learning?\n\n",
    "The history of artificial intelligence begins with",
]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]):
    """Print aligned table from list of dicts with same keys."""
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{"key": c}]) for c in cols]
    widths = [max(w, len(c)) for w, c in zip(widths, cols)]
    sep    = "  ".join("-" * w for w in widths)
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/eval_results",
                   help="Output directory for JSON report and per-model samples")
    p.add_argument("--n-chunks",     type=int, default=300,
                   help="Held-out FineWeb chunks for perplexity")
    p.add_argument("--n-alpaca",     type=int, default=100,
                   help="Held-out Alpaca examples for distilled model eval")
    p.add_argument("--max-gen",      type=int, default=80,
                   help="Max new tokens per generation sample")
    p.add_argument("--skip-gen",     action="store_true",
                   help="Skip generation (faster eval)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    CKPTS = {
        "128M-base":       ("outputs/aetherforge_fineweb_128M/final/model.pt", "128M", None,    None),
        "128M-distilled":  ("outputs/aetherforge_distill_5k/final/model.pt",   "128M",
                            "outputs/aetherforge_distill_5k/final/config.json", "qwen"),
        "1B-base":         ("outputs/aetherforge_1B_pretrain/final/model.pt",   "1B",  None,    None),
    }

    # Validate paths
    for name, (ckpt, _, _, _) in CKPTS.items():
        if not Path(ckpt).exists():
            print(f"WARNING: {name} checkpoint not found at {ckpt}")

    # ── Held-out data (load once) ────────────────────────────────────────
    print("\n── Loading held-out evaluation data ──────────────────────────────")
    fw_chunks   = get_fineweb_chunks(n=args.n_chunks, skip=15000)
    alpaca_test = get_alpaca_test(n=args.n_alpaca, skip=3000)

    results = {}
    gen_samples = {}

    # ── Evaluate each model ──────────────────────────────────────────────
    for name, (ckpt_path, config, cfg_json, tok_type) in CKPTS.items():
        if not Path(ckpt_path).exists():
            print(f"\nSkipping {name}: checkpoint missing")
            continue

        print(f"\n── {name} ──────────────────────────────────────────────────────")
        torch.cuda.empty_cache()

        try:
            if tok_type == "qwen":
                model, tokenizer = load_distill_model(ckpt_path, cfg_json)
            else:
                model    = load_charmodel(ckpt_path, config)
                tokenizer = None

            n_params = sum(p.numel() for p in model.parameters())
            vram     = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
            print(f"  Params: {n_params/1e6:.1f}M   VRAM: {vram:.2f} GB")

            # Perplexity eval
            if tok_type == "qwen":
                print(f"  Evaluating on {len(alpaca_test)} held-out Alpaca examples ...")
                ppl_result = perplexity_on_instructions(model, tokenizer, alpaca_test)
                ppl_result["eval_set"] = "alpaca-test"
            else:
                print(f"  Evaluating on {len(fw_chunks)} held-out FineWeb chunks ...")
                ppl_result = perplexity_on_chunks(model, fw_chunks)
                ppl_result["eval_set"] = "fineweb-held-out"

            ppl_result["n_params_M"] = round(n_params / 1e6, 1)
            results[name] = ppl_result
            print(f"  CE = {ppl_result['mean_ce_nats']} nats  "
                  f"ppl = {ppl_result['perplexity']}  "
                  f"({ppl_result['eval_seconds']}s)")

            # Generation samples
            if not args.skip_gen:
                print(f"  Generating {len(CHAT_PROMPTS)} samples ...")
                prompts = INSTRUCTION_PROMPTS if tok_type == "qwen" else CHAT_PROMPTS
                try:
                    completions = generate_completions(
                        model, prompts, tokenizer=tokenizer,
                        max_new_tokens=args.max_gen,
                    )
                    gen_samples[name] = completions
                except Exception as e:
                    print(f"  Generation error: {e}")
                    gen_samples[name] = [f"ERROR: {e}"] * len(CHAT_PROMPTS)

            del model
            if tok_type == "qwen":
                del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n══════════════════════════════════════════════════════════════════")
    print("  EVALUATION SUMMARY")
    print("══════════════════════════════════════════════════════════════════")
    table_rows = []
    for name, r in results.items():
        table_rows.append({
            "model":      name,
            "params_M":   r.get("n_params_M", "?"),
            "eval_set":   r.get("eval_set", "?"),
            "CE_nats":    r.get("mean_ce_nats", "?"),
            "perplexity": r.get("perplexity", "?"),
            "n_chunks":   r.get("n_chunks") or r.get("n_examples", "?"),
            "eval_sec":   r.get("eval_seconds", "?"),
        })
    print_table(table_rows)

    # ── Generation samples ───────────────────────────────────────────────
    if gen_samples:
        print("\n\n══════════════════════════════════════════════════════════════════")
        print("  GENERATION SAMPLES  (temperature=0.7, top_p=0.9, max_tokens=80)")
        print("══════════════════════════════════════════════════════════════════")
        for i, prompt in enumerate(CHAT_PROMPTS):
            print(f"\nPrompt {i+1}: {prompt[:80]}")
            print("─" * 64)
            for name, comps in gen_samples.items():
                c = comps[i] if i < len(comps) else "(missing)"
                c_short = c[:200].replace("\n", " ")
                print(f"  [{name:20s}]  {c_short}")

    # ── Save JSON ────────────────────────────────────────────────────────
    report = {
        "metrics":     results,
        "generations": {
            model_name: [
                {"prompt": CHAT_PROMPTS[i], "completion": c}
                for i, c in enumerate(comps)
            ]
            for model_name, comps in gen_samples.items()
        },
    }
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
