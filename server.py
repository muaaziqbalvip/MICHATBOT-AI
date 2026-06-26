"""
MI AI SERVER — SIMPLE & FAST
Urdu/English Bilingual, Mental Health Aware, Tokenizer Correct
"""

import os
import sys
import time
import threading
import re
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID = os.getenv("MODEL_ID", "miai-v1")
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-v1")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("MIAI_API_KEY", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

# CPU threads — GitHub Actions has 2 vCPU
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    "miai-v1": {"family": "qwen", "max_tokens": 180, "greedy": True, "quantize": True},
    "miai-v2": {"family": "qwen", "max_tokens": 260, "greedy": True, "quantize": True},
    "miai-v3": {"family": "smollm2", "max_tokens": 260, "greedy": False, "quantize": True},
    "miai-v4": {"family": "phi2", "max_tokens": 320, "greedy": False, "quantize": True},
}

CFG = CONFIG.get(MODEL_ID, CONFIG["miai-v1"])

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — WARM, MENTAL HEALTH AWARE, BILINGUAL
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are MI AI, MuslimIslam Organization's assistant. "
    "Reply in the same language the user used (Urdu, Roman Urdu, or English). "
    "Be warm, patient, and understanding. If the user seems upset or stressed, "
    "acknowledge their feeling gently before answering. Be accurate, helpful, "
    "clear. Never be cold or robotic. Keep answers focused and short."
)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading {MODEL_ID} ({CFG['family']})...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # INT8 quantization — 2-4x faster on CPU
    if CFG.get("quantize", True):
        try:
            print("⚡ Quantizing to INT8...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ INT8 quantization done")
        except Exception as e:
            print(f"⚠️ Quantization skipped: {e}")

    print(f"✅ {MODEL_ID} loaded!\n")

except Exception as e:
    print(f"❌ Load failed: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# EOS TOKENS
# ═══════════════════════════════════════════════════════════════════════════

def get_eos_ids(tok):
    ids = set()
    if tok.eos_token_id:
        ids.add(tok.eos_token_id)
    for s in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if tid and tid != tok.unk_token_id:
                ids.add(tid)
        except:
            pass
    return list(ids)

EOS_IDS = get_eos_ids(tokenizer)

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title=f"MI AI — {MODEL_ID}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS

# ═══════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════

def check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid key")

# ═══════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════

LEAKS = [
    r"<\|im_start\|>.*",
    r"<\|im_end\|>.*",
    r"\n?User:.*",
    r"\n?Assistant:.*",
]

def clean(text: str) -> str:
    for pat in LEAKS:
        text = re.split(pat, text, maxsplit=1, flags=re.DOTALL)[0]
    return text.strip()

# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt(messages):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        msgs.append({"role": m.role, "content": m.content})

    try:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
    except:
        pass

    # Fallback
    if CFG["family"] in ("qwen", "smollm2"):
        prompt = ""
        for msg in msgs:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
    else:
        prompt = ""
        for msg in msgs:
            if msg["role"] in ("system", "user"):
                prompt += f"Instruct: {msg['content']}\n"
            else:
                prompt += f"Output: {msg['content']}\n"
        prompt += "Output:"
    return prompt

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "languages": ["Urdu", "Roman Urdu", "English"],
    }

@app.get("/api/status/{model_name}")
def status(model_name: str):
    return {"model": model_name, "online": (model_name == MODEL_ID)}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start = time.time()

        # Build & tokenize
        prompt = build_prompt(req.messages)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048, padding=False
        )

        max_new = min(req.max_tokens or MAX_TOKENS, CFG["max_tokens"])
        temp = req.temperature if req.temperature else TEMPERATURE

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": EOS_IDS if EOS_IDS else tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }

        if CFG.get("greedy", False):
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = temp > 0.01
            gen_kwargs["temperature"] = max(temp, 0.01)
            gen_kwargs["top_p"] = 0.92

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = clean(reply)

        if not reply:
            reply = "Maaf kijiye, samajh nahi aaya. Dobara poochain?"

        elapsed = round(time.time() - start, 2)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_len + len(new_tokens),
                "latency_seconds": elapsed,
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
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════
# NGROK
# ═══════════════════════════════════════════════════════════════════════════

def start_ngrok():
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(PORT)
        print(f"\n🌐 Public URL: {tunnel.public_url}\n")
    except Exception as e:
        print(f"❌ Ngrok: {e}")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    print(f"🚀 Starting on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
