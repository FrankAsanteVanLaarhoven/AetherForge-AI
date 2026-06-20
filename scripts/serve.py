"""
AetherForge inference server — FastAPI, true token streaming, OpenAI-compatible API.

Serves AetherForge (any size, optional checkpoint) or Qwen2.5-VL-7B (4-bit + LoRA).
Both backends expose the same endpoint surface so clients can switch with one flag.

Endpoints:
    # AetherForge / native
    GET  /health                  — liveness check
    GET  /info                    — model name, params, config, VRAM
    POST /generate                — {"prompt": "...", "max_tokens": 200}
    POST /stream                  — same, SSE token stream  (true per-token)
    POST /chat                    — {"messages": [...]}
    POST /chat/stream             — chat with SSE stream

    # OpenAI-compatible  (drop-in for clients targeting api.openai.com)
    GET  /v1/models               — model listing
    POST /v1/completions          — text completion (OpenAI Completions format)
    POST /v1/chat/completions     — chat completion  (streaming supported)

Usage:
    # AetherForge 128M with no checkpoint (random weights — smoke-test)
    conda run -n ml-torch python scripts/serve.py

    # With a trained checkpoint
    conda run -n ml-torch python scripts/serve.py \\
        --model aetherforge \\
        --checkpoint outputs/aetherforge_pretrain/final/model.pt \\
        --config 128M

    # AetherForge with Qwen tokenizer (proper BPE, not char-level)
    conda run -n ml-torch python scripts/serve.py \\
        --tokenizer Qwen/Qwen2.5-VL-7B-Instruct \\
        --config 128M

    # Long-context AetherForge  (4× context extension)
    conda run -n ml-torch python scripts/serve.py \\
        --config 1B-8k \\
        --checkpoint outputs/aetherforge_pretrain/final/model.pt

    # Qwen2.5-VL-7B + LoRA
    conda run -n ml-torch python scripts/serve.py \\
        --model qwen \\
        --lora-path outputs/qwen25_vl_lora/final

    # cURL examples
    curl http://localhost:8000/generate \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "The transformer architecture"}'

    curl http://localhost:8000/stream \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Explain sparse MoE in one sentence"}' \\
        --no-buffer

    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "aetherforge", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Server deps missing: pip install fastapi uvicorn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Globals (set during startup)
# ---------------------------------------------------------------------------

_model        = None
_tokenizer    = None   # HF tokenizer (optional)
_model_name   = "unknown"
_model_info   = {}
_device       = "cuda" if torch.cuda.is_available() else "cpu"
_backend      = "aetherforge"    # "aetherforge" | "qwen"
_start_time   = time.time()


# ---------------------------------------------------------------------------
# Request / response schemas — native API
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:             str
    max_tokens:         int   = Field(200,  ge=1, le=4096)
    temperature:        float = Field(0.8,  ge=0.0, le=2.0)
    top_p:              float = Field(0.9,  ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0,  ge=1.0, le=2.0)
    stop:               Optional[list[str]] = None

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages:           list[ChatMessage]
    max_tokens:         int   = Field(200,  ge=1, le=4096)
    temperature:        float = Field(0.8,  ge=0.0, le=2.0)
    top_p:              float = Field(0.9,  ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0,  ge=1.0, le=2.0)
    stop:               Optional[list[str]] = None

class GenerateResponse(BaseModel):
    text:         str
    prompt_tokens: int
    output_tokens: int
    elapsed_ms:    int
    model:         str
    tokens_per_sec: float


# ---------------------------------------------------------------------------
# OpenAI-compatible schemas
# ---------------------------------------------------------------------------

class V1CompletionRequest(BaseModel):
    model:       str   = "aetherforge"
    prompt:      str
    max_tokens:  int   = 200
    temperature: float = 0.8
    top_p:       float = 0.9
    stream:      bool  = False
    stop:        Optional[list[str]] = None

class V1ChatMessage(BaseModel):
    role:    str
    content: str

class V1ChatRequest(BaseModel):
    model:       str   = "aetherforge"
    messages:    list[V1ChatMessage]
    max_tokens:  int   = 200
    temperature: float = 0.8
    top_p:       float = 0.9
    stream:      bool  = False
    stop:        Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _encode(text: str, vocab_size: int = 32000) -> list[int]:
    if _tokenizer is not None:
        return _tokenizer.encode(text, add_special_tokens=False)
    return [ord(c) % vocab_size for c in text]


def _decode(token_ids: list[int]) -> str:
    if _tokenizer is not None:
        return _tokenizer.decode(token_ids, skip_special_tokens=True)
    return "".join(chr(min(t, 127)) for t in token_ids)


def _eos_id() -> int | None:
    if _tokenizer is not None:
        return _tokenizer.eos_token_id
    return None


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_aetherforge(checkpoint: str | None, config: str, tokenizer_id: str | None):
    global _model, _tokenizer, _model_name, _model_info, _backend
    from aetherforge.model import AetherForge, MODEL_CONFIGS

    _backend = "aetherforge"

    if tokenizer_id:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        vocab_size = len(_tokenizer)
        print(f"Tokenizer: {tokenizer_id}  (vocab {vocab_size})")
    else:
        vocab_size = None

    cfg = dict(MODEL_CONFIGS.get(config, MODEL_CONFIGS["128M"]))
    if vocab_size:
        cfg["vocab_size"] = vocab_size

    model = AetherForge(**cfg).to(_device)

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=_device, weights_only=True)
        model.load_state_dict(state)
        print(f"Weights: {checkpoint}")
    else:
        print("No checkpoint — random weights (smoke-test only)")

    model.eval()
    _model      = model
    _model_name = f"AetherForge-{config}"
    _model_info = {
        "id":     _model_name,
        "params": model.param_count(),
        "config": config,
        "rope_scale": cfg.get("rope_scale", 1.0),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if _device == "cuda" else 0,
        "device": _device,
        "tokenizer": tokenizer_id or "char-level",
    }


def load_qwen(lora_path: str | None):
    global _model, _tokenizer, _model_name, _model_info, _backend
    from transformers import AutoTokenizer, BitsAndBytesConfig, \
        Qwen2_5_VLForConditionalGeneration

    _backend  = "qwen"
    MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
    quant     = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading {MODEL_ID} in 4-bit ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map="auto",
        trust_remote_code=True,
    )
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        print(f"LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    _tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model      = model
    _model_name = "Qwen2.5-VL-7B" + (" + LoRA" if lora_path else "")
    _model_info = {
        "id":     _model_name,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "device": _device,
        "tokenizer": MODEL_ID,
    }
    print(f"{_model_name} ready.")


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def _gen_aetherforge(prompt: str, max_tokens: int, temperature: float,
                     top_p: float, rep_penalty: float,
                     stop: list[str] | None = None) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, output_tokens)."""
    vocab_size = _model.embedding.num_embeddings
    ids        = _encode(prompt, vocab_size)
    n_prompt   = len(ids)
    input_ids  = torch.tensor([ids], device=_device, dtype=torch.long)

    out = _model.generate(
        input_ids, max_new_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        eos_token_id=_eos_id(), repetition_penalty=rep_penalty,
    )
    new_ids = out[0, n_prompt:].tolist()
    text    = _decode(new_ids)

    if stop:
        for s in stop:
            if s in text:
                text = text[:text.index(s)]

    return text, n_prompt, len(new_ids)


def _gen_qwen(prompt: str, max_tokens: int, temperature: float,
              top_p: float, stop: list[str] | None = None) -> tuple[str, int, int]:
    messages = [{"role": "user", "content": prompt}]
    tmpl     = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs   = _tokenizer(tmpl, return_tensors="pt").to(_model.device)
    n_prompt = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=(temperature > 0), temperature=max(temperature, 1e-6),
            top_p=top_p, pad_token_id=_tokenizer.eos_token_id,
        )
    new_ids = out[0, n_prompt:].tolist()
    text    = _tokenizer.decode(new_ids, skip_special_tokens=True)

    if stop:
        for s in stop:
            if s in text:
                text = text[:text.index(s)]

    return text, n_prompt, len(new_ids)


def generate_sync(prompt: str, max_tokens: int = 200,
                  temperature: float = 0.8, top_p: float = 0.9,
                  rep_penalty: float = 1.0,
                  stop: list[str] | None = None) -> tuple[str, int, int]:
    if _backend == "qwen":
        return _gen_qwen(prompt, max_tokens, temperature, top_p, stop)
    return _gen_aetherforge(prompt, max_tokens, temperature, top_p, rep_penalty, stop)


def messages_to_prompt(messages: list) -> str:
    """Flatten chat messages into a single prompt string (AetherForge fallback)."""
    parts = []
    for m in messages:
        role    = m.role if hasattr(m, "role") else m["role"]
        content = m.content if hasattr(m, "content") else m["content"]
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts) + "\nASSISTANT:"


# ---------------------------------------------------------------------------
# True per-token streaming — AetherForge
# ---------------------------------------------------------------------------

async def _stream_aetherforge(prompt: str, max_tokens: int, temperature: float,
                               top_p: float, rep_penalty: float,
                               stop: list[str] | None) -> AsyncGenerator[str, None]:
    """Yield decoded tokens one by one from AetherForge."""
    vocab_size = _model.embedding.num_embeddings
    ids        = _encode(prompt, vocab_size)
    input_ids  = torch.tensor([ids], device=_device, dtype=torch.long)
    eos        = _eos_id()
    generated  = ""

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = _model(input_ids)[:, -1, :]

        if rep_penalty != 1.0:
            for tok in input_ids[0].unique():
                logits[0, tok] /= rep_penalty

        if temperature == 0.0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_tok = torch.multinomial(sorted_probs, 1)
            next_tok = sorted_idx.gather(-1, next_tok)

        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        token_text = _decode([next_tok.item()])
        generated += token_text

        if stop and any(s in generated for s in stop):
            break

        yield f"data: {json.dumps({'token': token_text, 'id': next_tok.item()})}\n\n"
        await asyncio.sleep(0)   # yield control to event loop between tokens

        if eos is not None and next_tok.item() == eos:
            break

    yield "data: [DONE]\n\n"


async def _stream_qwen(prompt: str, max_tokens: int, temperature: float,
                       top_p: float) -> AsyncGenerator[str, None]:
    """Yield tokens from Qwen via HF TextIteratorStreamer in a thread."""
    from transformers import TextIteratorStreamer

    messages = [{"role": "user", "content": prompt}]
    tmpl     = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs   = _tokenizer(tmpl, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(
        _tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    def _run():
        _model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=(temperature > 0), temperature=max(temperature, 1e-6),
            top_p=top_p, pad_token_id=_tokenizer.eos_token_id,
            streamer=streamer,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    for text_chunk in streamer:
        if text_chunk:
            yield f"data: {json.dumps({'token': text_chunk})}\n\n"
            await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


async def stream_tokens(prompt: str, max_tokens: int, temperature: float,
                        top_p: float, rep_penalty: float,
                        stop: list[str] | None) -> AsyncGenerator[str, None]:
    if _backend == "qwen":
        async for chunk in _stream_qwen(prompt, max_tokens, temperature, top_p):
            yield chunk
    else:
        async for chunk in _stream_aetherforge(
            prompt, max_tokens, temperature, top_p, rep_penalty, stop
        ):
            yield chunk


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "AetherForge API",
    description = "LLM inference server — AetherForge or Qwen2.5-VL backend",
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Native endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     _model_name,
        "device":    _device,
        "uptime_s":  int(time.time() - _start_time),
        "timestamp": int(time.time()),
    }


@app.get("/info")
def info():
    return _model_info


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    t0   = time.time()
    text, n_prompt, n_out = generate_sync(
        req.prompt, req.max_tokens, req.temperature, req.top_p,
        req.repetition_penalty, req.stop,
    )
    elapsed = time.time() - t0
    return GenerateResponse(
        text          = text,
        prompt_tokens = n_prompt,
        output_tokens = n_out,
        elapsed_ms    = int(elapsed * 1000),
        model         = _model_name,
        tokens_per_sec= round(n_out / max(elapsed, 1e-6), 1),
    )


@app.post("/stream")
async def stream_endpoint(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return StreamingResponse(
        stream_tokens(req.prompt, req.max_tokens, req.temperature,
                      req.top_p, req.repetition_penalty, req.stop),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat", response_model=GenerateResponse)
def chat_endpoint(req: ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    prompt = messages_to_prompt(req.messages)
    t0     = time.time()
    text, n_prompt, n_out = generate_sync(
        prompt, req.max_tokens, req.temperature, req.top_p,
        req.repetition_penalty, req.stop,
    )
    elapsed = time.time() - t0
    return GenerateResponse(
        text          = text,
        prompt_tokens = n_prompt,
        output_tokens = n_out,
        elapsed_ms    = int(elapsed * 1000),
        model         = _model_name,
        tokens_per_sec= round(n_out / max(elapsed, 1e-6), 1),
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    prompt = messages_to_prompt(req.messages)
    return StreamingResponse(
        stream_tokens(prompt, req.max_tokens, req.temperature,
                      req.top_p, req.repetition_penalty, req.stop),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── OpenAI-compatible endpoints ───────────────────────────────────────────

@app.get("/v1/models")
def v1_models():
    return {
        "object": "list",
        "data": [{
            "id":       _model_name,
            "object":   "model",
            "created":  int(_start_time),
            "owned_by": "aetherforge",
            **_model_info,
        }],
    }


@app.post("/v1/completions")
async def v1_completions(req: V1CompletionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    if req.stream:
        async def _sse():
            cid = f"cmpl-{int(time.time())}"
            async for chunk in stream_tokens(
                req.prompt, req.max_tokens, req.temperature, req.top_p, 1.0, req.stop
            ):
                if chunk.strip() == "data: [DONE]":
                    yield f"data: [DONE]\n\n"
                    return
                token = json.loads(chunk[6:]).get("token", "")
                payload = {
                    "id": cid, "object": "text_completion",
                    "model": _model_name,
                    "choices": [{"text": token, "index": 0,
                                 "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    t0   = time.time()
    text, n_prompt, n_out = generate_sync(
        req.prompt, req.max_tokens, req.temperature, req.top_p, 1.0, req.stop
    )
    return {
        "id":      f"cmpl-{int(t0)}",
        "object":  "text_completion",
        "created": int(t0),
        "model":   _model_name,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     n_prompt,
            "completion_tokens": n_out,
            "total_tokens":      n_prompt + n_out,
        },
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: V1ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    prompt = messages_to_prompt(req.messages)

    if req.stream:
        async def _sse():
            cid = f"chatcmpl-{int(time.time())}"
            async for chunk in stream_tokens(
                prompt, req.max_tokens, req.temperature, req.top_p, 1.0, req.stop
            ):
                if chunk.strip() == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                    return
                token = json.loads(chunk[6:]).get("token", "")
                payload = {
                    "id": cid, "object": "chat.completion.chunk",
                    "model": _model_name,
                    "choices": [{"delta": {"content": token},
                                 "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    t0   = time.time()
    text, n_prompt, n_out = generate_sync(
        prompt, req.max_tokens, req.temperature, req.top_p, 1.0, req.stop
    )
    return {
        "id":      f"chatcmpl-{int(t0)}",
        "object":  "chat.completion",
        "created": int(t0),
        "model":   _model_name,
        "choices": [{
            "message":       {"role": "assistant", "content": text},
            "index":         0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     n_prompt,
            "completion_tokens": n_out,
            "total_tokens":      n_prompt + n_out,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BANNER = """
  ╔═══════════════════════════════════════════╗
  ║          AetherForge Inference API        ║
  ║    v0.1.0  ·  Newcastle University        ║
  ╚═══════════════════════════════════════════╝
"""


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="AetherForge inference server (OpenAI-compatible API)",
    )
    p.add_argument("--model",      choices=["aetherforge", "qwen"], default="aetherforge",
                   help="Backend model to serve.")
    p.add_argument("--config",     default="128M",
                   help="AetherForge size config. Choices: "
                        "128M, 1B, 7B, 13B, 1B-8k, 1B-32k, 7B-32k")
    p.add_argument("--checkpoint", default=None,
                   help="AetherForge .pt checkpoint file.")
    p.add_argument("--tokenizer",  default=None,
                   help="HF tokenizer id (e.g. Qwen/Qwen2.5-VL-7B-Instruct). "
                        "Uses char-level fallback if omitted.")
    p.add_argument("--lora-path",  default=None,
                   help="LoRA adapter directory (Qwen backend only).")
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--port",       type=int, default=8000)
    p.add_argument("--workers",    type=int, default=1,
                   help="Uvicorn workers (>1 requires --checkpoint; "
                        "each worker loads a separate model copy).")
    args = p.parse_args()

    print(BANNER)

    if args.model == "aetherforge":
        load_aetherforge(args.checkpoint, args.config, args.tokenizer)
    else:
        load_qwen(args.lora_path)

    print(f"\n  Endpoints:")
    print(f"    http://{args.host}:{args.port}/docs          — Swagger UI")
    print(f"    http://{args.host}:{args.port}/generate      — POST generation")
    print(f"    http://{args.host}:{args.port}/stream        — POST SSE streaming")
    print(f"    http://{args.host}:{args.port}/v1/chat/completions  — OpenAI-compat")
    print()

    uvicorn.run(app, host=args.host, port=args.port,
                workers=args.workers, log_level="warning")


if __name__ == "__main__":
    main()
