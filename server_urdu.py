"""
MI AI URDU SERVER — miai-urdu
Dedicated Urdu Text Model

Base: Qwen2.5-1.5B-Instruct (multilingual, handles Urdu script reliably
via its tokenizer/chat template) wrapped with a strict Urdu-only system
prompt so replies always come back in Urdu script, regardless of how
the user writes (Urdu, Roman Urdu, or English).
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
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID = "miai-urdu"
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-urdu")
PORT = int(os.getenv("PORT", "8004"))
API_KEY = os.getenv("MIAI_API_KEY", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "280"))
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading {MODEL_ID} (Qwen2.5-1.5B-Instruct, Urdu-primed)...")

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

    print(f"✅ {MODEL_ID} loaded successfully!\n")

except Exception as e:
    print(f"❌ Load failed: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# EOS TOKENS
# ═══════════════════════════════════════════════════════════════════════════

def get_eos_ids(tok):
    ids = set()
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, list):
            ids.update(tok.eos_token_id)
        else:
            ids.add(tok.eos_token_id)
    for s in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
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

app = FastAPI(title="MI AI — miai-urdu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    r"<\|endoftext\|>.*",
    r"\n?User:.*",
    r"\n?Assistant:.*",
]

def clean(text: str) -> str:
    for pat in LEAKS:
        text = re.split(pat, text, maxsplit=1, flags=re.DOTALL)[0]
    return text.strip()

# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING — strict Urdu-script system prompt
# ═══════════════════════════════════════════════════════════════════════════

URDU_SYSTEM_PROMPT = (
    "آپ ایم آئی اے آئی کے اردو زبان کے ماہر اسسٹنٹ ہیں۔ ہمیشہ خالص اور رواں اردو "
    "رسم الخط میں جواب دیں، چاہے صارف رومن اردو یا انگریزی میں سوال کرے۔ صرف اس وقت "
    "انگریزی استعمال کریں جب کوئی تکنیکی اصطلاح، کوڈ، یا برانڈ کا نام ہو۔"
)

def build_prompt(messages):
    msgs = [{"role": "system", "content": URDU_SYSTEM_PROMPT}]
    msgs += [{"role": m.role, "content": m.content} for m in messages]

    try:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
    except:
        pass

    prompt = f"System: {URDU_SYSTEM_PROMPT}\n"
    for msg in msgs[1:]:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "purpose": "dedicated Urdu text model"}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start = time.time()

        prompt = build_prompt(req.messages)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048, padding=False
        )

        max_new = min(req.max_tokens or MAX_TOKENS, 320)
        temp = req.temperature if req.temperature is not None else TEMPERATURE

        gen_kwargs = {
            "max_new_tokens": max_new,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": EOS_IDS if EOS_IDS else tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "use_cache": True,
            "do_sample": True,
            "temperature": temp,
            "top_p": 0.9,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = clean(reply)

        if not reply:
            reply = "معذرت، دوبارہ کوشش کریں۔"

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

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ═══════════════════════════════════════════════════════════════════════════
# NGROK & MAIN
# ═══════════════════════════════════════════════════════════════════════════

def start_ngrok():
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(PORT)
        print(f"\n🌐 Public URL: {tunnel.public_url}\n")
    except Exception as e:
        print(f"❌ Ngrok Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    print(f"🚀 Starting miai-urdu server on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
