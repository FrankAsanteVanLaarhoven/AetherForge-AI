"""
scripts/chat.py — interactive CLI chat with AetherForge or Qwen2.5-VL.

Works in two modes:
  1. Direct model mode: loads the model locally (uses GPU).
  2. Server mode: connects to a running serve.py instance via HTTP.

Usage:
    # Direct — AetherForge (random weights if no checkpoint)
    conda run -n ml-torch python scripts/chat.py

    # Direct — AetherForge with trained checkpoint
    conda run -n ml-torch python scripts/chat.py \\
        --checkpoint outputs/aetherforge_pretrain/final/model.pt \\
        --config 128M

    # Direct — AetherForge with Qwen tokenizer (real BPE)
    conda run -n ml-torch python scripts/chat.py \\
        --tokenizer Qwen/Qwen2.5-VL-7B-Instruct --config 128M

    # Direct — long-context 1B (NTK-aware RoPE, ~8K context)
    conda run -n ml-torch python scripts/chat.py \\
        --config 1B-8k \\
        --checkpoint outputs/aetherforge_pretrain/final/model.pt

    # Direct — Qwen2.5-VL-7B (requires ~6GB VRAM)
    conda run -n ml-torch python scripts/chat.py --model qwen

    # Server mode — connect to a running serve.py
    conda run -n ml-torch python scripts/chat.py \\
        --server http://localhost:8000

Commands during chat:
    /help      Show this list
    /clear     Reset conversation history
    /config    Print model / generation settings
    /temp N    Set temperature (0 = greedy)
    /top_p N   Set top-p nucleus cutoff
    /tokens N  Set max new tokens per turn
    /exit      Quit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colours — disabled if not a tty or on Windows
USE_COLOR = sys.stdout.isatty() and os.name != "nt"
RESET  = "\033[0m"  if USE_COLOR else ""
BOLD   = "\033[1m"  if USE_COLOR else ""
DIM    = "\033[2m"  if USE_COLOR else ""
GOLD   = "\033[93m" if USE_COLOR else ""
BLUE   = "\033[94m" if USE_COLOR else ""
GREEN  = "\033[92m" if USE_COLOR else ""
RED    = "\033[91m" if USE_COLOR else ""
PURPLE = "\033[95m" if USE_COLOR else ""

BANNER = f"""
{GOLD}{BOLD}  AetherForge Chat  v0.1.0{RESET}
{DIM}  /help for commands · /exit to quit{RESET}
"""

HELP_TEXT = f"""
{BOLD}Commands:{RESET}
  {BLUE}/clear{RESET}      Reset conversation history
  {BLUE}/config{RESET}     Print current model + generation settings
  {BLUE}/temp N{RESET}     Set sampling temperature  (0 = greedy argmax)
  {BLUE}/top_p N{RESET}    Set top-p nucleus cutoff  (0.0–1.0)
  {BLUE}/tokens N{RESET}   Set max new tokens per response
  {BLUE}/rep N{RESET}      Set repetition penalty  (1.0 = disabled)
  {BLUE}/exit{RESET}       Quit
"""


# ---------------------------------------------------------------------------
# Server-mode client
# ---------------------------------------------------------------------------

def chat_via_server(server_url: str, args):
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        pass

    url = server_url.rstrip("/")

    # Check health
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            info = json.loads(r.read())
        model_name = info.get("model", "unknown")
    except Exception as e:
        print(f"{RED}Cannot reach server at {url}: {e}{RESET}")
        print("Start one with:  make serve")
        sys.exit(1)

    print(BANNER)
    print(f"{DIM}  Connected to {url}  ·  model: {model_name}{RESET}\n")

    temperature  = args.temperature
    top_p        = args.top_p
    max_tokens   = args.max_tokens
    rep_penalty  = args.repetition_penalty
    history: list[dict] = []

    while True:
        try:
            user_input = input(f"{GOLD}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lstrip("/").split()
            name = cmd[0].lower()
            if name in ("exit", "quit", "q"):
                print("Bye.")
                break
            elif name == "clear":
                history.clear()
                print(f"{DIM}History cleared.{RESET}")
            elif name == "config":
                print(f"{DIM}  model={model_name}  temp={temperature}  "
                      f"top_p={top_p}  max_tokens={max_tokens}  rep={rep_penalty}{RESET}")
            elif name == "temp" and len(cmd) == 2:
                temperature = float(cmd[1]); print(f"{DIM}temperature={temperature}{RESET}")
            elif name == "top_p" and len(cmd) == 2:
                top_p = float(cmd[1]); print(f"{DIM}top_p={top_p}{RESET}")
            elif name == "tokens" and len(cmd) == 2:
                max_tokens = int(cmd[1]); print(f"{DIM}max_tokens={max_tokens}{RESET}")
            elif name == "rep" and len(cmd) == 2:
                rep_penalty = float(cmd[1]); print(f"{DIM}rep_penalty={rep_penalty}{RESET}")
            elif name == "help":
                print(HELP_TEXT)
            else:
                print(f"{DIM}Unknown command. /help for list.{RESET}")
            continue

        history.append({"role": "user", "content": user_input})
        payload = json.dumps({
            "messages": history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{url}/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            elapsed = time.time() - t0
            text = resp.get("text", "").strip()
            tps  = resp.get("tokens_per_sec", 0)
            history.append({"role": "assistant", "content": text})
            print(f"\n{BLUE}{BOLD}AetherForge:{RESET} {text}")
            print(f"{DIM}  ({resp.get('output_tokens', '?')} tokens · "
                  f"{tps} tok/s · {int(elapsed*1000)} ms){RESET}\n")
        except Exception as e:
            print(f"{RED}Request failed: {e}{RESET}\n")


# ---------------------------------------------------------------------------
# Direct model mode
# ---------------------------------------------------------------------------

def chat_direct(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ──────────────────────────────────────────────────────
    if args.model == "qwen":
        from transformers import AutoTokenizer, BitsAndBytesConfig, \
            Qwen2_5_VLForConditionalGeneration
        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        quant    = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        print(f"Loading {MODEL_ID} in 4-bit ...")
        model    = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=quant, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model_name = "Qwen2.5-VL-7B"
        backend    = "qwen"

    else:
        from aetherforge.model import AetherForge, MODEL_CONFIGS

        tokenizer = None
        vocab_size = None
        if args.tokenizer:
            from transformers import AutoTokenizer as HFTok
            tokenizer  = HFTok.from_pretrained(args.tokenizer, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            vocab_size = len(tokenizer)

        cfg = dict(MODEL_CONFIGS.get(args.config, MODEL_CONFIGS["128M"]))
        if vocab_size:
            cfg["vocab_size"] = vocab_size
        model = AetherForge(**cfg).to(device)

        if args.checkpoint and Path(args.checkpoint).exists():
            state = torch.load(args.checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Weights: {args.checkpoint}")
        else:
            print(f"{DIM}No checkpoint — random weights{RESET}")

        model.eval()
        model_name = f"AetherForge-{args.config}"
        backend    = "aetherforge"

    vram = (f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
            if device == "cuda" else "CPU")
    print(BANNER)
    print(f"{DIM}  model: {model_name}  |  {vram}  |  device: {device}{RESET}\n")

    # ── Generation settings ─────────────────────────────────────────────
    temperature  = args.temperature
    top_p        = args.top_p
    max_tokens   = args.max_tokens
    rep_penalty  = args.repetition_penalty
    history: list[dict] = []

    def _encode(text: str) -> list[int]:
        if tokenizer:
            return tokenizer.encode(text, add_special_tokens=False)
        vs = model.embedding.num_embeddings
        return [ord(c) % vs for c in text]

    def _decode(ids: list[int]) -> str:
        if tokenizer:
            return tokenizer.decode(ids, skip_special_tokens=True)
        return "".join(chr(min(t, 127)) for t in ids)

    def _eos():
        return tokenizer.eos_token_id if tokenizer else None

    def respond(prompt: str) -> tuple[str, float]:
        if backend == "qwen":
            messages = history[:]
            tmpl     = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs   = tokenizer(tmpl, return_tensors="pt").to(model.device)
            n_prompt = inputs["input_ids"].shape[-1]
            t0       = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=max(temperature, 1e-6),
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out[0, n_prompt:].tolist()
            text    = tokenizer.decode(new_ids, skip_special_tokens=True)
            elapsed = time.time() - t0
            return text.strip(), len(new_ids) / max(elapsed, 1e-6)

        # AetherForge
        ids       = _encode(prompt)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        t0        = time.time()
        out       = model.generate(
            input_ids, max_new_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
            eos_token_id=_eos(), repetition_penalty=rep_penalty,
        )
        new_ids = out[0, len(ids):].tolist()
        text    = _decode(new_ids)
        elapsed = time.time() - t0
        return text.strip(), len(new_ids) / max(elapsed, 1e-6)

    # ── REPL ────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{GOLD}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd  = user_input.lstrip("/").split()
            name = cmd[0].lower()
            if name in ("exit", "quit", "q"):
                print("Bye.")
                break
            elif name == "clear":
                history.clear()
                print(f"{DIM}History cleared.{RESET}")
            elif name == "config":
                print(f"{DIM}  model={model_name}  temp={temperature}  "
                      f"top_p={top_p}  tokens={max_tokens}  rep={rep_penalty}{RESET}")
            elif name == "temp" and len(cmd) == 2:
                temperature = float(cmd[1]); print(f"{DIM}temperature → {temperature}{RESET}")
            elif name == "top_p" and len(cmd) == 2:
                top_p = float(cmd[1]); print(f"{DIM}top_p → {top_p}{RESET}")
            elif name == "tokens" and len(cmd) == 2:
                max_tokens = int(cmd[1]); print(f"{DIM}max_tokens → {max_tokens}{RESET}")
            elif name == "rep" and len(cmd) == 2:
                rep_penalty = float(cmd[1]); print(f"{DIM}rep_penalty → {rep_penalty}{RESET}")
            elif name == "help":
                print(HELP_TEXT)
            else:
                print(f"{DIM}Unknown command. /help for list.{RESET}")
            continue

        history.append({"role": "user", "content": user_input})
        # Build flat prompt for AetherForge
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) + "\nASSISTANT:"

        try:
            text, tps = respond(prompt)
        except Exception as e:
            print(f"{RED}Generation error: {e}{RESET}\n")
            continue

        history.append({"role": "assistant", "content": text})
        print(f"\n{BLUE}{BOLD}AetherForge:{RESET} {text}")
        print(f"{DIM}  ({tps:.0f} tok/s){RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Interactive chat CLI for AetherForge / Qwen2.5-VL",
    )
    p.add_argument("--model",      choices=["aetherforge", "qwen"], default="aetherforge")
    p.add_argument("--config",     default="128M",
                   help="AetherForge config. "
                        "Options: 128M, 1B, 7B, 13B, 1B-8k, 1B-32k, 7B-32k")
    p.add_argument("--checkpoint", default=None,
                   help="Path to AetherForge .pt checkpoint file.")
    p.add_argument("--tokenizer",  default=None,
                   help="HF tokenizer for AetherForge "
                        "(e.g. Qwen/Qwen2.5-VL-7B-Instruct).")
    p.add_argument("--server",     default=None,
                   help="URL of running serve.py  "
                        "(e.g. http://localhost:8000). Skips local model load.")
    p.add_argument("--temperature",        type=float, default=0.7)
    p.add_argument("--top-p",             type=float, default=0.9, dest="top_p")
    p.add_argument("--max-tokens",        type=int,   default=200, dest="max_tokens")
    p.add_argument("--repetition-penalty",type=float, default=1.1, dest="repetition_penalty")
    args = p.parse_args()

    if args.server:
        chat_via_server(args.server, args)
    else:
        chat_direct(args)


if __name__ == "__main__":
    main()
