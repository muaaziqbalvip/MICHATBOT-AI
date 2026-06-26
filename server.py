"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    MI AI SERVER v3.1 — COMPLETE                           ║
║                                                                            ║
║  Fast Urdu/English AI Model Server for GitHub Actions (CPU-only)          ║
║  Models: Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B, Phi-2 2.7B             ║
║                                                                            ║
║  FEATURES:                                                                 ║
║  • INT8 quantization → 2-4x faster on CPU                                 ║
║  • Live date/time injection → models know current date                    ║
║  • Warm, patient tone → not robotic                                       ║
║  • Bilingual (Urdu/English) → same language reply                         ║
║  • Streaming support → response starts instantly                          ║
║  • Health check + model status endpoints                                  ║
║                                                                            ║
║  DEPLOYMENT:                                                               ║
║  - Set MODEL_ID env var: miai-v1, miai-v2, miai-v3, or miai-v4            ║
║  - Set NGROK_AUTH_TOKEN for public tunnel                                 ║
║  - Optional: MIAI_API_KEY for auth, TZ_OFFSET_HOURS for timezone          ║
║  - Runs on port 8000 (configurable via PORT env var)                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
import re
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional

# ============================================================================
# IMPORTS — AI/Web
# ============================================================================
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model & paths
MODEL_ID = os.getenv("MODEL_ID", "miai-v1")
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-v1")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("MIAI_API_KEY", "")

# Temperature & token limits
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

# Server's real clock (fresh date/time injected every request)
# Default: Pakistan Standard Time (UTC+5)
TZ_OFFSET_HOURS = float(os.getenv("TZ_OFFSET_HOURS", "5"))
LOCAL_TZ = timezone(timedelta(hours=TZ_OFFSET_HOURS))

# CPU threads — GitHub Actions gives 2 vCPUs
# Pinning to 2 prevents thread contention that slows inference down.
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

# Ngrok token for public tunnel
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
if not NGROK_TOKEN:
    print("❌ CRITICAL: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

# ============================================================================
# MODEL FAMILY CONFIGS
# ============================================================================
# Each model family (Qwen, SmolLM2, Phi-2) needs different settings:
# - max_new_tokens_cap: max tokens this model can generate
# - default_max_tokens: sensible default for normal Q&A
# - repetition_penalty: how much to discourage repeating text
# - greedy: True = deterministic (fast, good for small models)
#           False = sampling (slower, better variety for large models)
# - quantize: use INT8 quantization on Linear layers

FAMILY_CONFIG = {
    "miai-v1": {  # Qwen2.5-0.5B — smallest, fastest
        "family": "qwen",
        "max_new_tokens_cap": 180,
        "default_max_tokens": 100,
        "repetition_penalty": 1.15,
        "greedy": True,
        "quantize": True,
    },
    "miai-v2": {  # Qwen2.5-1.5B — small, balanced
        "family": "qwen",
        "max_new_tokens_cap": 260,
        "default_max_tokens": 140,
        "repetition_penalty": 1.1,
        "greedy": True,
        "quantize": True,
    },
    "miai-v3": {  # SmolLM2-1.7B — medium, multilingual
        "family": "smollm2",
        "max_new_tokens_cap": 260,
        "default_max_tokens": 140,
        "repetition_penalty": 1.1,
        "greedy": False,
        "quantize": True,
    },
    "miai-v4": {  # Phi-2 2.7B — large, best quality
        "family": "phi2",
        "max_new_tokens_cap": 320,
        "default_max_tokens": 160,
        "repetition_penalty": 1.05,
        "greedy": False,
        "quantize": True,
    },
}

CFG = FAMILY_CONFIG.get(MODEL_ID, FAMILY_CONFIG["miai-v1"])
_env_max_tokens = os.getenv("MAX_TOKENS")
MAX_TOKENS = int(_env_max_tokens) if _env_max_tokens else CFG["default_max_tokens"]

# ============================================================================
# SYSTEM PROMPT — BILINGUAL, WARM, AWARE OF REAL TIME
# ============================================================================

SYSTEM_PROMPT_BASE = (
    "You are MI AI, assistant of MuslimIslam Organization. "
    "Reply in the same language the user used (Roman Urdu, Urdu, or English). "
    "Be accurate, clear, helpful, including with code and technical questions. "
    "\n\n"
    "TONE: Talk like a thoughtful, patient person — not a robot or machine. "
    "Before answering, acknowledge how the user might be feeling if it seems "
    "like they're frustrated, stressed, or upset. Keep that warm tone throughout. "
    "Don't be cold, dismissive, or purely mechanical. One genuine sentence first, "
    "then solve the problem. Keep answers focused, avoid repeating yourself."
)

def current_time_block() -> str:
    """Returns a text block with current date/time for the system prompt."""
    now = datetime.now(LOCAL_TZ)
    formatted_date = now.strftime("%A, %d %B %Y")
    formatted_time = now.strftime("%I:%M %p")
    return (
        f"[SERVER CLOCK] Current date & time (Pakistan Standard Time, live): "
        f"{formatted_date}, {formatted_time}. "
        "If user asks 'what time/date is it', use this as your answer — "
        "do NOT say you don't know, do NOT guess a different date."
    )

# ============================================================================
# DIRECT TIME/DATE ANSWERING — BYPASS MODEL FOR INSTANT CORRECT ANSWER
# ============================================================================
# When user just asks "what time is it", the server answers immediately
# from its real clock, not the model (which has no clock). This guarantees
# a correct answer, and is also instant.

TIME_QUERY_PATTERNS = [
    r"\b(time|date|day)\b.*\b(today|right now|abhi|is waqt)\b",
    r"\b(abhi|aaj|is waqt)\b.*\b(time|waqt|tareekh|date|din)\b",
    r"^\s*(what'?s the time|what time is it|kya time hai|time kya hai|aaj ki tareekh|kya tareekh hai)\s*\??\s*$",
    r"\bcurrent (time|date)\b",
]
_TIME_QUERY_RE = re.compile("|".join(TIME_QUERY_PATTERNS), re.IGNORECASE)

def try_direct_time_answer(user_text: str) -> Optional[str]:
    """If user is asking about current time/date, return it immediately."""
    if not user_text or not _TIME_QUERY_RE.search(user_text):
        return None
    now = datetime.now(LOCAL_TZ)
    formatted_date = now.strftime("%A, %d %B %Y")
    formatted_time = now.strftime("%I:%M %p")
    return (
        f"Abhi {formatted_time} ho raha hai, {formatted_date} "
        f"(Pakistan Standard Time)."
    )

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================

print(f"\n{'='*80}")
print(f"🔄 Booting {MODEL_ID} ({CFG['family']}) from {MODEL_PATH}")
print(f"   CPU threads: {CPU_THREADS} (GitHub Actions runner)")
print(f"   Timezone: UTC+{TZ_OFFSET_HOURS}")
print(f"{'='*80}\n")

try:
    print(f"📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    print(f"   ✓ Chat template present: {has_chat_template}")

    print(f"🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ─────────────────────────────────────────────────────────────────────
    # INT8 DYNAMIC QUANTIZATION (2-4x speedup on CPU)
    # ─────────────────────────────────────────────────────────────────────
    # Quantizing Linear layers to INT8 makes them 4x smaller in memory,
    # which means faster matrix multiplications on CPU (bandwidth-bound).
    # This is the single biggest lever for CPU inference speed.
    if CFG.get("quantize", True):
        try:
            print("⚡ Applying INT8 dynamic quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("   ✓ Linear layers quantized to INT8 (4x smaller weights)")
        except Exception as qe:
            print(f"   ⚠️ Quantization skipped ({type(qe).__name__}) — running in fp32")

    print(f"✅ {MODEL_ID} loaded successfully!\n")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    sys.exit(1)

# ============================================================================
# BUILD EOS TOKEN LIST
# ============================================================================
# Different models have different stop tokens. Collecting all of them
# ensures the model stops generating at the right place.

def build_eos_ids(tok):
    """Collect all possible end-of-sequence token IDs."""
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
print(f"   EOS token IDs: {EOS_IDS}\n")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title=f"MI AI — {MODEL_ID} (v3.1)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class Message(BaseModel):
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "system"
    content: str  # The text

class ChatRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""
    model: str = MODEL_ID
    messages: List[Message]
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    stream: Optional[bool] = False

# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_auth(authorization: Optional[str]):
    """Check Bearer token if API key is configured."""
    if not API_KEY:
        return  # No key configured, auth disabled
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ============================================================================
# OUTPUT CLEANING
# ============================================================================
# Small models sometimes leak role tags or start hallucinating a new turn.
# Clean those out so the response looks natural.

ROLE_LEAK_PATTERNS = [
    r"<\|im_start\|>.*",
    r"<\|im_end\|>.*",
    r"<\|endoftext\|>.*",
    r"\n?User:.*",
    r"\n?Assistant:.*",
    r"\n?System:.*",
    r"\n?Human:.*",
]

def clean_output(text: str) -> str:
    """Remove role-tag leaks and garbage from model output."""
    cleaned = text
    for pat in ROLE_LEAK_PATTERNS:
        cleaned = re.split(pat, cleaned, maxsplit=1, flags=re.DOTALL)[0]
    return cleaned.strip()

# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_prompt(messages):
    """Build the full prompt with system message + live time block."""
    live_system = SYSTEM_PROMPT_BASE + "\n\n" + current_time_block()
    formatted = [{"role": "system", "content": live_system}]
    for m in messages:
        formatted.append({"role": m.role, "content": m.content})

    # Try to use tokenizer's built-in chat template
    try:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True
            )
        raise ValueError("no chat template")
    except Exception:
        # Fallback: build prompt manually per model family
        if CFG["family"] in ("qwen", "smollm2"):
            # ChatML format: <|im_start|>role\ncontent<|im_end|>
            prompt = ""
            for msg in formatted:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        else:
            # Phi-2 format: Instruct: ... Output: ...
            prompt = ""
            for msg in formatted:
                if msg["role"] in ("system", "user"):
                    prompt += f"Instruct: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Output: {msg['content']}\n"
            prompt += "Output:"
        return prompt

# ============================================================================
# GENERATION KWARGS BUILDER
# ============================================================================

def build_gen_kwargs(max_new, temp):
    """Build kwargs dict for model.generate() based on model family."""
    gen_kwargs = dict(
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=EOS_IDS if EOS_IDS else tokenizer.eos_token_id,
        repetition_penalty=CFG["repetition_penalty"],
        use_cache=True,  # Enable KV cache for faster decoding
    )

    if CFG.get("greedy", False):
        # Greedy: deterministic, fast, good for small models
        gen_kwargs["do_sample"] = False
    else:
        # Sampling: more varied, slower, good for larger models
        gen_kwargs["do_sample"] = temp > 0.01
        gen_kwargs["temperature"] = max(temp, 0.01)
        gen_kwargs["top_p"] = 0.92

    return gen_kwargs

# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": MODEL_ID,
        "family": CFG["family"],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "quantized": CFG.get("quantize", True),
        "cpu_threads": CPU_THREADS,
        "eos_ids": EOS_IDS,
        "server_time": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p"),
        "timezone": f"UTC+{TZ_OFFSET_HOURS}",
        "languages": ["Urdu", "Roman Urdu", "English"],
    }

@app.get("/api/status/{model_name}")
def status(model_name: str):
    """Check if a specific model is online."""
    online = (model_name == MODEL_ID)
    return {"model": model_name, "online": online}

@app.get("/v1/models")
def list_models():
    """List all available models."""
    return {
        "object": "list",
        "data": [
            {"id": "miai-v1", "description": "Qwen2.5-0.5B — Ultra fast"},
            {"id": "miai-v2", "description": "Qwen2.5-1.5B — Balanced"},
            {"id": "miai-v3", "description": "SmolLM2-1.7B — Multilingual"},
            {"id": "miai-v4", "description": "Phi-2 2.7B — Best quality"},
        ]
    }

# ============================================================================
# STREAMING RESPONSE GENERATOR
# ============================================================================

def sse_stream(inputs, gen_kwargs):
    """Generate response as server-sent events (stream=true)."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread_kwargs = dict(gen_kwargs)
    thread_kwargs["streamer"] = streamer

    # Run generation in background thread
    thread = threading.Thread(target=lambda: model.generate(**inputs, **thread_kwargs))
    thread.start()

    chunk_id = f"chatcmpl-{MODEL_ID}-{int(time.time())}"
    raw_buffer = ""
    emitted_len = 0
    stop_emitting = False

    # Stream tokens as they're generated
    for new_text in streamer:
        raw_buffer += new_text
        if stop_emitting:
            continue

        cleaned = clean_output(raw_buffer)
        if len(cleaned) < len(raw_buffer.strip()):
            # Role-leak pattern triggered — stop emitting
            stop_emitting = True

        piece = cleaned[emitted_len:]
        emitted_len = len(cleaned)

        if piece:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n"

    thread.join()

    # Send final chunk
    done_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"

# ============================================================================
# MAIN CHAT ENDPOINT
# ============================================================================

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    """Main chat completion endpoint (OpenAI-compatible API)."""
    check_auth(authorization)

    try:
        start_time = time.time()

        # ─────────────────────────────────────────────────────────────────
        # QUICK TIME/DATE ANSWER
        # ─────────────────────────────────────────────────────────────────
        # If user is just asking what time/date it is, answer instantly
        # from server clock instead of running the model.
        last_user_msg = next(
            (m.content for m in reversed(req.messages) if m.role == "user"), ""
        )
        direct_answer = try_direct_time_answer(last_user_msg)

        if direct_answer and not req.stream:
            elapsed = round(time.time() - start_time, 3)
            return {
                "id": f"chatcmpl-{MODEL_ID}-{int(time.time())}",
                "object": "chat.completion",
                "model": MODEL_ID,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": elapsed,
                    "tokens_per_second": 0,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": direct_answer},
                        "finish_reason": "stop",
                    }
                ],
            }

        # ─────────────────────────────────────────────────────────────────
        # BUILD PROMPT & TOKENIZE
        # ─────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────
        # STREAMING MODE
        # ─────────────────────────────────────────────────────────────────
        if req.stream:
            return StreamingResponse(
                sse_stream(inputs, gen_kwargs),
                media_type="text/event-stream",
            )

        # ─────────────────────────────────────────────────────────────────
        # NON-STREAMING MODE
        # ─────────────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = clean_output(raw_reply)

        # Fallback if nothing was generated
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
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in {MODEL_ID}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"{MODEL_ID} Error: {str(e)}"
        )

# ============================================================================
# NGROK TUNNEL (PUBLIC URL)
# ============================================================================

def start_ngrok():
    """Start ngrok tunnel for public access."""
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(PORT)
        print(f"\n🌐 [{MODEL_ID}] Public URL: {tunnel.public_url}\n")
    except exception.PyngrokNgrokError as e:
        print(f"❌ Ngrok Error: {e}")
        sys.exit(1)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Start ngrok in background
    ngrok_thread = threading.Thread(target=start_ngrok)
    ngrok_thread.daemon = True
    ngrok_thread.start()

    print(f"🚀 Starting {MODEL_ID} on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
