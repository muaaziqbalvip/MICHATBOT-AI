"""
MI AI Server — Urdu Language Support v2.1
Supports: Urdu, Roman Urdu, English - Dynamic language detection
"""

import os
import sys
import time
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_ID",   "miai-v1")
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-v1")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
PORT       = int(os.getenv("PORT", "8000"))

NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
if not NGROK_TOKEN:
    print("❌ CRITICAL: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

# ─── URDU SYSTEM PROMPT ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MI AI - MuslimIslam Organization's official assistant.

LANGUAGE HANDLING (BILKUL ZARURI):
1. Agar user Urdu/Roman Urdu mein likhe → HAMESHA Roman Urdu mein jawab do
2. Agar English mein likhe → English mein jawab do
3. Mixed language? → User ki primary language mein jawab do

STYLE:
- Short answers, no unnecessary text
- Direct aur helpful
- Islamic values in all responses
- Urdu speakers ko comfort dena

EXAMPLES:
User: "Aap kaun ho?"
Answer: "Main MI AI hoon, MuslimIslam Organization ka assistant. Aap mujhe kuch pooch sakte ho."

User: "Islam mein namaz ki ahammiyat?"
Answer: "Namaz Islam ka aik aham stoon hai. Har Muslim par 5 waqt namaz zaruri hai..."

User: "Who are you?"
Answer: "I am MI AI, assistant of MuslimIslam Organization. How can I help you?"

CRITICAL: Kabhi confuse mat ho Urdu text se. Seedha samajho aur jawab do!"""

# ─── Load Model ────────────────────────────────────────────────────────────
print(f"\n🔄 Booting {MODEL_ID} from {MODEL_PATH}...")

try:
    print(f"📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # CRITICAL FIX: Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add Urdu-friendly settings
    tokenizer.padding_side = "left"  # Left padding for generation
    
    print(f"🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"✅ {MODEL_ID} loaded successfully!\n")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    sys.exit(1)

# ─── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(title=f"MI AI — {MODEL_ID} Engine")

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

# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "languages": ["Urdu", "Roman Urdu", "English"]
    }

# ─── Models List ────────────────────────────────────────────────────────────
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

# ─── Main Chat Endpoint (URDU-OPTIMIZED) ──────────────────────────────────
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        start_time = time.time()

        # Build formatted messages
        formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.messages:
            formatted.append({"role": m.role, "content": m.content})

        # Apply chat template
        try:
            prompt = tokenizer.apply_chat_template(
                formatted,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            prompt = ""
            for msg in formatted:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            prompt += "Assistant:"

        # Tokenize with proper Urdu handling
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False  # No padding for generation
        )

        # Generate with Urdu-optimized settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 0.01),
                do_sample=req.temperature > 0,
                top_p=0.95,                    # Nucleus sampling for diversity
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,        # Prevent repetition
                no_repeat_ngram_size=3,        # Block n-gram repeats
                early_stopping=False,
            )

        # Extract only new tokens
        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elapsed = round(time.time() - start_time, 2)

        return {
            "id": f"chatcmpl-{MODEL_ID}-{int(time.time())}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_len + len(new_tokens),
                "latency_seconds": elapsed,
            },
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }]
        }

    except Exception as e:
        print(f"Error: {str(e)}")
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
