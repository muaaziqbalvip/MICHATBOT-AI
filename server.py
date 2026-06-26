"""
MI AI Server — Urdu Language Support v3.0 (FAST ENGINE)
Supports: Urdu, Roman Urdu, English - Dynamic language detection

WHAT CHANGED IN v3.0 (SPEED + QUALITY UPGRADE)
────────────────────────────────────────────────────────────────────────────
GitHub Actions runners (`ubuntu-latest`) only give 2 vCPUs and NO GPU. That
is a hard ceiling — no code change makes Phi-2 run like it's on an A100.
But a LOT of speed was being left on the table by the old code. Fixed here:

1. INT8 DYNAMIC QUANTIZATION (the single biggest win)
   torch.quantization.quantize_dynamic on all Linear layers. On CPU this
   typically gives a 2-4x speedup on the matmul-heavy forward pass with
   only a tiny quality cost — because CPU inference is bandwidth-bound,
   not compute-bound, and int8 weights are 4x smaller to move through
   cache. This is the correct lever to pull, not a placebo.

2. THREAD COUNT MATCHED TO THE RUNNER
   torch.set_num_threads(2) — GitHub Actions gives 2 vCPUs. PyTorch
   defaults to detecting ALL logical cores including ones you don't have
   exclusive access to, which causes thread contention and SLOWS things
   down, not speeds them up. Pinning this to the real core count fixed a
   measurable amount of latency variance.

3. GREEDY DECODING FOR SMALL MODELS (miai-v1, miai-v2)
   do_sample=False = fewer ops per token AND more reliable/coherent output
   on small models, which tend to wander under sampling anyway. v3/v4
   keep light sampling since they're big enough to use it well.

4. SHORTER, TIGHTER SYSTEM PROMPT FOR ALL MODELS
   Every token in the system prompt is a token the model has to process
   on EVERY single request before it even starts answering. The old
   "full" prompt was processed fresh every request. Trimmed it down
   while keeping the bilingual behavior and Islamic-context identity.

5. KV-CACHE EXPLICITLY ON
   `use_cache=True` stops a few wasted recomputations during decode.

6. MAX_TOKENS DEFAULT LOWERED PER MODEL
   Long answers are simply slower — that's tokens-per-second math, not
   a bug. Default output length trimmed sensibly per model so a normal
   question doesn't generate 150-300 tokens unless it actually needs to.
   (Still fully overridable via the `max_tokens` field in the request.)

7. STREAMING SUPPORT (NEW)
   Added a `/v1/chat/completions` stream=True path using TextIteratorStreamer.
   This doesn't make total generation faster, but it makes the response feel
   close to instant because the user sees the first words almost immediately
   instead of waiting for the entire reply to finish.

8. MENTAL-HEALTH / TONE UPGRADE (as requested)
   Every model now has a standing instruction to respond with patience,
   warmth, and emotional awareness — not just dry facts — and to never be
   dismissive, rude, or robotic in Urdu or English.

Everything else (auth, /health, /api/status, model routing, the OpenAI-
style response shape) is untouched on purpose so your dashboard and Vercel
gateway keep working exactly as before.
"""

import os
import sys
import time
import threading
import re
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_ID    = os.getenv("MODEL_ID", "miai-v1")
MODEL_PATH  = os.getenv("MODEL_PATH", "./model_files/miai-v1")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
PORT        = int(os.getenv("PORT", "8000"))
API_KEY     = os.getenv("MIAI_API_KEY", "")  # set this in your env / Vercel config

# GitHub Actions ubuntu-latest = 2 vCPUs. Pin threads to match — letting
# PyTorch over-subscribe threads on a shared runner makes things SLOWER,
# not faster, due to contention.
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
if not NGROK_TOKEN:
    print("❌ CRITICAL: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

# ─── MODEL FAMILY DETECTION ────────────────────────────────────────────────
FAMILY_CONFIG = {
    "miai-v1": {  # Qwen2.5-0.5B
        "family": "qwen",
        "max_new_tokens_cap": 180,
        "default_max_tokens": 100,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 0,
        "greedy": True,          # small model -> greedy = faster + more coherent
        "quantize": True,
    },
    "miai-v2": {  # Qwen2.5-1.5B
        "family": "qwen",
        "max_new_tokens_cap": 260,
        "default_max_tokens": 140,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "greedy": True,
        "quantize": True,
    },
    "miai-v3": {  # SmolLM2-1.7B
        "family": "smollm2",
        "max_new_tokens_cap": 260,
        "default_max_tokens": 140,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "greedy": False,
        "quantize": True,
    },
    "miai-v4": {  # Phi-2 2.7B
        "family": "phi2",
        "max_new_tokens_cap": 320,
        "default_max_tokens": 160,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 0,
        "greedy": False,
        "quantize": True,
    },
}
CFG = FAMILY_CONFIG.get(MODEL_ID, FAMILY_CONFIG["miai-v1"])

_env_max_tokens = os.getenv("MAX_TOKENS")
MAX_TOKENS = int(_env_max_tokens) if _env_max_tokens else CFG["default_max_tokens"]

# ─── SYSTEM PROMPT (trimmed — every token here is paid on every request) ───
# Short, bilingual, and includes a tone instruction (patience / emotional
# awareness) as requested, without ballooning the prompt back up.
SYSTEM_PROMPT = (
    "You are MI AI, MuslimIslam Organization's assistant. "
    "Reply in the same language the user used (Roman Urdu, Urdu, or English). "
    "Be accurate, clear, and helpful, including with code and technical questions. "
    "Be warm and patient: never dismissive or robotic. If the user sounds upset "
    "or distressed, respond gently and supportively before answering. "
    "Keep answers focused and avoid unnecessary repetition."
)

# ─── Load Model ────────────────────────────────────────────────────────────
print(f"\n🔄 Booting {MODEL_ID} ({CFG['family']}) from {MODEL_PATH}...")
print(f"   CPU threads pinned to: {CPU_THREADS}")

try:
    print(f"📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    print(f"   Chat template present: {has_chat_template}")

    print(f"🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ── INT8 DYNAMIC QUANTIZATION (the real speed lever on CPU) ──────────
    # Quantizing the Linear layers is what actually moves the needle for
    # CPU-only inference: smaller weights -> better cache use -> faster
    # matmuls. Wrapped in try/except because quantize_dynamic support can
    # vary slightly by torch build; if it fails we fall back to fp32
    # rather than crashing the whole engine.
    if CFG.get("quantize", True):
        try:
            print("⚡ Applying INT8 dynamic quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ Quantization applied — Linear layers now INT8")
        except Exception as qe:
            print(f"⚠️ Quantization skipped ({qe}) — continuing in fp32")

    print(f"✅ {MODEL_ID} loaded successfully!\n")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    sys.exit(1)

# ─── Build stop-token-id list ──────────────────────────────────────────────
def build_eos_ids(tok):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    candidates = ["<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"]
    for tok_str in candidates:
        try:
            tid = tok.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != tok.unk_token_id:
                ids.add(tid)
        except Exception:
            pass
    return list(ids)

EOS_IDS = build_eos_ids(tokenizer)
print(f"   EOS token ids in use: {EOS_IDS}")

# ─── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(title=f"MI AI — {MODEL_ID} Engine (Fast)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    stream: Optional[bool] = False

# ─── Auth helper ────────────────────────────────────────────────────────────
def check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ─── Output cleaning ────────────────────────────────────────────────────────
ROLE_LEAK_PATTERNS = [
    r"<\|im_start\|>.*", r"<\|im_end\|>.*", r"<\|endoftext\|>.*",
    r"\n?User:.*", r"\n?Assistant:.*", r"\n?System:.*",
    r"\n?Human:.*",
]

def clean_output(text: str) -> str:
    cleaned = text
    for pat in ROLE_LEAK_PATTERNS:
        cleaned = re.split(pat, cleaned, maxsplit=1, flags=re.DOTALL)[0]
    return cleaned.strip()

def build_prompt(messages):
    """Builds the model prompt with per-family fallback if no chat template."""
    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        formatted.append({"role": m.role, "content": m.content})

    try:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True
            )
        raise ValueError("no chat template")
    except Exception:
        if CFG["family"] in ("qwen", "smollm2"):
            prompt = ""
            for msg in formatted:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        else:
            prompt = ""
            for msg in formatted:
                if msg["role"] in ("system", "user"):
                    prompt += f"Instruct: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Output: {msg['content']}\n"
            prompt += "Output:"
        return prompt

def build_gen_kwargs(max_new, temp):
    gen_kwargs = dict(
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=EOS_IDS if EOS_IDS else tokenizer.eos_token_id,
        repetition_penalty=CFG["repetition_penalty"],
        use_cache=True,
    )
    if CFG.get("greedy", False):
        # Greedy decoding: fewer ops per step, more stable on small models.
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = temp > 0.01
        gen_kwargs["temperature"] = max(temp, 0.01)
        gen_kwargs["top_p"] = 0.92
    if CFG["no_repeat_ngram_size"] > 0:
        gen_kwargs["no_repeat_ngram_size"] = CFG["no_repeat_ngram_size"]
    return gen_kwargs

# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "family": CFG["family"],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "quantized": CFG.get("quantize", True),
        "cpu_threads": CPU_THREADS,
        "eos_ids": EOS_IDS,
        "languages": ["Urdu", "Roman Urdu", "English"],
    }

@app.get("/api/status/{model_name}")
def status(model_name: str):
    online = (model_name == MODEL_ID)
    return {"model": model_name, "online": online}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "miai-v1", "object": "model", "description": "Qwen2.5-0.5B — Ultra fast, Urdu support"},
            {"id": "miai-v2", "object": "model", "description": "Qwen2.5-1.5B — Balanced speed + quality"},
            {"id": "miai-v3", "object": "model", "description": "SmolLM2-1.7B — Smart multilingual"},
            {"id": "miai-v4", "object": "model", "description": "Phi-2 2.7B — Deep reasoning, best quality"},
        ]
    }

# ─── Streaming generator ───────────────────────────────────────────────────
def sse_stream(inputs, gen_kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread_kwargs = dict(gen_kwargs)
    thread_kwargs["streamer"] = streamer

    thread = threading.Thread(target=lambda: model.generate(**inputs, **thread_kwargs))
    thread.start()

    chunk_id = f"chatcmpl-{MODEL_ID}-{int(time.time())}"
    raw_buffer = ""
    emitted_len = 0
    stop_emitting = False

    for new_text in streamer:
        raw_buffer += new_text
        if stop_emitting:
            continue
        cleaned = clean_output(raw_buffer)
        if len(cleaned) < len(raw_buffer.strip()):
            # A role-leak pattern triggered a cut — emit what's new, then stop.
            stop_emitting = True
        piece = cleaned[emitted_len:]
        emitted_len = len(cleaned)
        if piece:
            payload = {
                "id": chunk_id, "object": "chat.completion.chunk", "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n"

    thread.join()
    done_payload = {
        "id": chunk_id, "object": "chat.completion.chunk", "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"

# ─── Main Chat Endpoint ─────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start_time = time.time()
        prompt = build_prompt(req.messages)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )

        max_new = min(req.max_tokens or MAX_TOKENS, CFG["max_new_tokens_cap"])
        temp = req.temperature if req.temperature is not None else TEMPERATURE
        gen_kwargs = build_gen_kwargs(max_new, temp)

        if req.stream:
            return StreamingResponse(
                sse_stream(inputs, gen_kwargs),
                media_type="text/event-stream",
            )

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = clean_output(raw_reply)

        if not reply:
            reply = "Maaf kijiye, mujhe samajh nahi aaya. Dobara poochain?"

        elapsed = round(time.time() - start_time, 2)
        n_new = len(new_tokens)
        tps = round(n_new / elapsed, 2) if elapsed > 0 else 0.0

        return {
            "id": f"chatcmpl-{MODEL_ID}-{int(time.time())}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": n_new,
                "total_tokens": input_len + n_new,
                "latency_seconds": elapsed,
                "tokens_per_second": tps,
            },
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in {MODEL_ID}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{MODEL_ID} Error: {str(e)}")

# ─── Ngrok Tunnel ──────────────────────────────────────────────────────────
def start_ngrok():
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(PORT)
        print(f"\n🌐 [{MODEL_ID}] Public URL: {tunnel.public_url}\n")
    except exception.PyngrokNgrokError as e:
        print(f"❌ Ngrok Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    print(f"🚀 Starting {MODEL_ID} on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
