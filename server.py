"""
MI AI Server — Urdu Language Support v2.2 (FIXED)
Supports: Urdu, Roman Urdu, English - Dynamic language detection

FIXES IN THIS VERSION:
- Per-model chat template handling (Qwen2.5 / SmolLM2 / Phi-2 have different formats)
- Removed conflicting generation params (repetition_penalty + no_repeat_ngram together
  was breaking small models)
- Proper EOS token list (model.generate needs ALL stop tokens, not just one)
- Output cleaning (strips leaked role-tags/garbage that small models echo back)
- Per-model system prompt sizing (0.5B model can't follow long bilingual instructions
  as well as Phi-2 can — so it gets a shorter, simpler version)
- API key auth actually enforced (was missing — your dashboard sends an Authorization
  header but old server.py never checked it)
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

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_ID    = os.getenv("MODEL_ID", "miai-v1")
MODEL_PATH  = os.getenv("MODEL_PATH", "./model_files/miai-v1")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
PORT        = int(os.getenv("PORT", "8000"))
API_KEY     = os.getenv("MIAI_API_KEY", "")  # set this in your env / Vercel config

NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
if not NGROK_TOKEN:
    print("❌ CRITICAL: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

# ─── MODEL FAMILY DETECTION ────────────────────────────────────────────────
# Different base models need different chat-template handling and different
# generation settings. This was the #1 reason for "pagal" / garbage output —
# one-size-fits-all settings do not work across Qwen / SmolLM2 / Phi-2.
FAMILY_CONFIG = {
    "miai-v1": {  # Qwen2.5-0.5B
        "family": "qwen",
        "max_new_tokens_cap": 200,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 0,   # OFF — combined with rep_penalty it breaks small models
        "system_prompt": "short",
    },
    "miai-v2": {  # Qwen2.5-1.5B
        "family": "qwen",
        "max_new_tokens_cap": 300,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "system_prompt": "full",
    },
    "miai-v3": {  # SmolLM2-1.7B
        "family": "smollm2",
        "max_new_tokens_cap": 300,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
        "system_prompt": "full",
    },
    "miai-v4": {  # Phi-2 2.7B
        "family": "phi2",
        "max_new_tokens_cap": 400,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 0,
        "system_prompt": "full",
    },
}
CFG = FAMILY_CONFIG.get(MODEL_ID, FAMILY_CONFIG["miai-v1"])

# ─── SYSTEM PROMPTS ─────────────────────────────────────────────────────────
# Small models (0.5B) get confused by long, complex bilingual rule-lists.
# Give them a short version. Bigger models get the full version.

SYSTEM_PROMPT_FULL = """You are MI AI - MuslimIslam Organization's official assistant.

LANGUAGE RULE:
- If the user writes in Urdu or Roman Urdu, reply in Roman Urdu.
- If the user writes in English, reply in English.
- Match whichever language the user used most.

STYLE:
- Keep answers short and direct.
- Stay helpful and respectful of Islamic values.

EXAMPLES:
User: "Aap kaun ho?"
Assistant: "Main MI AI hoon, MuslimIslam Organization ka assistant. Aap mujhe kuch pooch sakte ho."

User: "Islam mein namaz ki ahammiyat?"
Assistant: "Namaz Islam ka aik aham stoon hai. Har Muslim par 5 waqt namaz zaruri hai."

User: "Who are you?"
Assistant: "I am MI AI, assistant of the MuslimIslam Organization. How can I help you?"
"""

SYSTEM_PROMPT_SHORT = """You are MI AI, MuslimIslam Organization's assistant.
Reply in the same language the user used (Roman Urdu or English).
Keep answers short and direct.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_FULL if CFG["system_prompt"] == "full" else SYSTEM_PROMPT_SHORT

# ─── Load Model ────────────────────────────────────────────────────────────
print(f"\n🔄 Booting {MODEL_ID} ({CFG['family']}) from {MODEL_PATH}...")

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

    # Sanity check: does this tokenizer actually have a chat template?
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
    print(f"✅ {MODEL_ID} loaded successfully!\n")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    sys.exit(1)

# ─── Build stop-token-id list (CRITICAL FIX) ───────────────────────────────
# Old code only passed tokenizer.eos_token_id (a single id). Qwen/SmolLM2/Phi-2
# chat-formatted models often have ADDITIONAL turn-end tokens (e.g. <|im_end|>)
# that are NOT the same as the base eos_token. If you don't pass those too,
# the model keeps generating past the end of its real answer -> garbage tail.
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

# ─── Auth helper (was missing in old server.py — dashboard sends a Bearer
# token but nothing ever checked it) ────────────────────────────────────────
def check_auth(authorization: Optional[str]):
    if not API_KEY:
        return  # no key configured -> auth disabled, dev mode
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ─── Output cleaning (CRITICAL FIX) ────────────────────────────────────────
# Small/medium models frequently echo role tags or start hallucinating a new
# "User:" turn after their real answer. This strips that out so the dashboard
# doesn't show "ulta pulta" text.
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

# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "family": CFG["family"],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "eos_ids": EOS_IDS,
        "languages": ["Urdu", "Roman Urdu", "English"],
    }

# ─── Status endpoint (your dashboard polls /api/status/{model} — add this
# so the dashboard's online/offline badges actually work) ──────────────────
@app.get("/api/status/{model_name}")
def status(model_name: str):
    online = (model_name == MODEL_ID)
    return {"model": model_name, "online": online}

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

# ─── Main Chat Endpoint (FIXED) ────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start_time = time.time()

        formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.messages:
            formatted.append({"role": m.role, "content": m.content})

        # Apply chat template — with a per-family fallback that actually
        # matches each model's real expected format (old code's fallback
        # was a generic "User: / Assistant:" format that NONE of these
        # models were trained on, which is a big source of "pagal" replies).
        try:
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    formatted,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                raise ValueError("no chat template")
        except Exception:
            if CFG["family"] in ("qwen", "smollm2"):
                # ChatML format
                prompt = ""
                for msg in formatted:
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
            else:
                # Phi-2 style
                prompt = ""
                for msg in formatted:
                    if msg["role"] == "system":
                        prompt += f"Instruct: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"Instruct: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Output: {msg['content']}\n"
                prompt += "Output:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )

        max_new = min(req.max_tokens or MAX_TOKENS, CFG["max_new_tokens_cap"])
        temp = max(req.temperature if req.temperature is not None else TEMPERATURE, 0.01)

        gen_kwargs = dict(
            max_new_tokens=max_new,
            temperature=temp,
            do_sample=temp > 0.01,
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=EOS_IDS if EOS_IDS else tokenizer.eos_token_id,
            repetition_penalty=CFG["repetition_penalty"],
        )
        # Only add no_repeat_ngram_size if explicitly enabled for this model —
        # combining it with repetition_penalty on small models is what was
        # causing broken / nonsensical Urdu output.
        if CFG["no_repeat_ngram_size"] > 0:
            gen_kwargs["no_repeat_ngram_size"] = CFG["no_repeat_ngram_size"]

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = clean_output(raw_reply)

        if not reply:
            reply = "Maaf kijiye, mujhe samajh nahi aaya. Dobara poochain?" \
                if CFG["system_prompt"] == "short" else \
                "Sorry, I couldn't generate a clear response. Please rephrase your question."

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
