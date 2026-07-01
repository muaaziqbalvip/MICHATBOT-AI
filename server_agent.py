"""
MI AI AGENT SERVER — miai-agent
Online Task / Tool-Use Agent

Base: Qwen2.5-3B-Instruct (bigger than the call/chat models since
agentic reasoning needs more headroom).

Tool: free, keyless web search via the `ddgs` (DuckDuckGo Search) package
— no API key needed, fits the project's "free-tier only" philosophy.
The agent decides whether a query needs a live web lookup, fetches a
few results if so, then reasons over them to answer.
"""

import os
import sys
import time
import json
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

MODEL_ID = "miai-agent"
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-agent")
PORT = int(os.getenv("PORT", "8005"))
API_KEY = os.getenv("MIAI_API_KEY", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════════════
# WEB SEARCH TOOL (free, no API key — ddgs package)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from ddgs import DDGS
    SEARCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ddgs not available, online search disabled: {e}")
    SEARCH_AVAILABLE = False

def web_search(query: str, max_results: int = 4):
    if not SEARCH_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

NEEDS_SEARCH_HINTS = [
    "today", "aaj", "current", "latest", "abhi", "news", "khabar",
    "price", "qeemat", "weather", "mosam", "score", "result", "2025", "2026",
]

def looks_like_it_needs_search(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in NEEDS_SEARCH_HINTS)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading {MODEL_ID} (Qwen2.5-3B-Instruct, agent-tuned)...")

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

app = FastAPI(title="MI AI — miai-agent")

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

class AgentTaskRequest(BaseModel):
    task: str
    force_search: Optional[bool] = None
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
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = (
    "You are MI AI's online task agent. You can be given live web search "
    "results as context — use them to give an accurate, up-to-date answer. "
    "If no search results are provided, answer from your own knowledge and "
    "say so if you're not certain about recent events. Be concise and "
    "structured. Match the user's language (Urdu, Roman Urdu, or English)."
)

def build_prompt(messages, search_context: str = ""):
    msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if search_context:
        msgs.append({"role": "system", "content": f"Live web search results:\n{search_context}"})
    msgs += messages

    try:
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
    except:
        pass

    prompt = f"System: {AGENT_SYSTEM_PROMPT}\n"
    if search_context:
        prompt += f"System: Live web search results:\n{search_context}\n"
    for msg in msgs[len(msgs) - len(messages):]:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def generate(prompt: str, max_new: int, temp: float):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3072, padding=False
    )
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
    reply = clean(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return reply, input_len, len(new_tokens)

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "purpose": "online task / tool-use agent",
        "search_available": SEARCH_AVAILABLE,
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    """Plain chat-shaped endpoint so it also works through the normal gateway."""
    check_auth(authorization)
    try:
        start = time.time()
        last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

        search_context = ""
        if last_user_msg and looks_like_it_needs_search(last_user_msg):
            results = web_search(last_user_msg)
            if results:
                search_context = "\n".join(
                    f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results
                )

        prompt = build_prompt([{"role": m.role, "content": m.content} for m in req.messages], search_context)
        max_new = min(req.max_tokens or MAX_TOKENS, 450)
        temp = req.temperature if req.temperature is not None else TEMPERATURE
        reply, input_len, out_len = generate(prompt, max_new, temp)

        if not reply:
            reply = "..."

        elapsed = round(time.time() - start, 2)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": out_len,
                "total_tokens": input_len + out_len,
                "latency_seconds": elapsed,
            },
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
            ],
            "used_web_search": bool(search_context),
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/v1/agent/tasks")
def agent_task(req: AgentTaskRequest, authorization: Optional[str] = Header(None)):
    """Dedicated task endpoint: give it a single task string, it decides
    whether to search the web, then returns a structured answer."""
    check_auth(authorization)
    try:
        start = time.time()

        do_search = req.force_search if req.force_search is not None else looks_like_it_needs_search(req.task)
        search_context = ""
        results = []
        if do_search:
            results = web_search(req.task)
            if results:
                search_context = "\n".join(
                    f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results
                )

        prompt = build_prompt([{"role": "user", "content": req.task}], search_context)
        max_new = min(req.max_tokens or MAX_TOKENS, 450)
        temp = req.temperature if req.temperature is not None else TEMPERATURE
        reply, input_len, out_len = generate(prompt, max_new, temp)

        if not reply:
            reply = "..."

        elapsed = round(time.time() - start, 2)
        return {
            "id": f"agent-{int(time.time())}",
            "object": "agent.task",
            "model": MODEL_ID,
            "task": req.task,
            "used_web_search": do_search and bool(results),
            "sources": results,
            "answer": reply,
            "latency_seconds": elapsed,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": out_len,
                "total_tokens": input_len + out_len,
            },
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
    print(f"🚀 Starting miai-agent server on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
