"""
MI AI Gateway v2.0 — Vercel API (api/index.py)
4 models support: miai-v1, miai-v2, miai-v3, miai-v4
Auto-routing: model name se sahi GitHub server pe forward karta hai.
"""

import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="MI AI Gateway v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each model ka Vercel env se URL milta hai
MODEL_URLS = {
    "miai-v1": os.getenv("MIAI_V1_URL", ""),
    "miai-v2": os.getenv("MIAI_V2_URL", ""),
    "miai-v3": os.getenv("MIAI_V3_URL", ""),
    "miai-v4": os.getenv("MIAI_V4_URL", ""),
}

LOCK_CODE = "muaaz19720"

# ─── Admin Dashboard ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    model_cards = ""
    for mid, minfo in {
        "miai-v1": {"name": "Qwen2.5-0.5B", "badge": "⚡ Ultra Fast"},
        "miai-v2": {"name": "Qwen2.5-1.5B", "badge": "⚖️ Balanced"},
        "miai-v3": {"name": "SmolLM2-1.7B",  "badge": "🌍 Multilingual"},
        "miai-v4": {"name": "Phi-2 2.7B",    "badge": "🧠 Deep Reason"},
    }.items():
        model_cards += f"""
        <div class="model-card" id="card-{mid}">
          <div class="model-badge">{minfo['badge']}</div>
          <div class="model-title">{mid}</div>
          <div class="model-sub">{minfo['name']}</div>
          <div class="model-status" id="status-{mid}">⏳ Checking...</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MI AI — Admin Dashboard v2</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0a0f1e; color: #f0f4ff; min-height: 100vh; }}
    header {{ background: linear-gradient(135deg, #0d1b2a, #1a3a5c); padding: 20px 30px; border-bottom: 1px solid #1e3a5f; display: flex; align-items: center; gap: 15px; }}
    header h1 {{ font-size: 22px; color: #38bdf8; letter-spacing: 2px; }}
    header span {{ font-size: 12px; color: #64748b; background: #0f2744; padding: 4px 10px; border-radius: 20px; }}
    .container {{ max-width: 900px; margin: 30px auto; padding: 0 20px; }}
    .lock-card {{ background: #111827; border: 1px solid #1f2d40; border-radius: 14px; padding: 30px; text-align: center; max-width: 380px; margin: 0 auto 30px; }}
    .lock-card h2 {{ color: #38bdf8; margin-bottom: 20px; }}
    input {{ width: 100%; padding: 12px; margin: 8px 0; background: #1e293b; border: 1px solid #334155; color: white; border-radius: 8px; font-size: 15px; outline: none; }}
    input:focus {{ border-color: #38bdf8; }}
    .btn {{ width: 100%; padding: 13px; background: #0284c7; border: none; color: white; font-size: 15px; border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 8px; transition: 0.2s; }}
    .btn:hover {{ background: #0369a1; }}
    .btn-green {{ background: #059669; }} .btn-green:hover {{ background: #047857; }}
    .models-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
    .model-card {{ background: #111827; border: 1px solid #1f2d40; border-radius: 12px; padding: 18px; text-align: center; transition: 0.2s; }}
    .model-card:hover {{ border-color: #38bdf8; transform: translateY(-2px); }}
    .model-badge {{ font-size: 11px; color: #94a3b8; background: #1e293b; padding: 3px 9px; border-radius: 12px; display: inline-block; margin-bottom: 8px; }}
    .model-title {{ font-size: 18px; font-weight: bold; color: #38bdf8; }}
    .model-sub {{ font-size: 12px; color: #64748b; margin: 4px 0 10px; }}
    .model-status {{ font-size: 13px; font-weight: 600; }}
    .online {{ color: #34d399; }} .offline {{ color: #ef4444; }}
    #controlSection {{ display: none; }}
    #keyBox {{ background: #0f172a; border: 1px dashed #34d399; border-radius: 8px; padding: 15px; margin-top: 15px; font-family: monospace; font-size: 13px; word-break: break-all; color: #34d399; display: none; }}
    .section-title {{ font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin: 24px 0 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>⚡ MI AI Dashboard</h1>
    <span>v2.0 — 4 Models Active</span>
  </header>
  <div class="container">
    <div class="lock-card">
      <div id="lockSection">
        <h2>🔐 System Lock</h2>
        <input type="password" id="lockInput" placeholder="Security Code" onkeydown="if(event.key==='Enter')unlock()">
        <button class="btn" onclick="unlock()">Unlock Dashboard</button>
      </div>
    </div>

    <div id="controlSection">
      <div class="section-title">📡 Live Models Status</div>
      <div class="models-grid">{model_cards}</div>

      <div class="lock-card" style="margin-top:0">
        <h2 style="font-size:16px;margin-bottom:12px">🔑 Generate API Key</h2>
        <select id="modelSel" style="width:100%;padding:10px;background:#1e293b;border:1px solid #334155;color:white;border-radius:8px;margin:8px 0">
          <option value="miai-v1">miai-v1 — Qwen 0.5B (Fast)</option>
          <option value="miai-v2">miai-v2 — Qwen 1.5B (Balanced)</option>
          <option value="miai-v3">miai-v3 — SmolLM2 1.7B</option>
          <option value="miai-v4">miai-v4 — Phi-2 2.7B (Best)</option>
        </select>
        <button class="btn btn-green" onclick="genKey()">Generate Key</button>
        <div id="keyBox"></div>
      </div>
    </div>
  </div>

  <script>
    function unlock() {{
      if(document.getElementById('lockInput').value === '{LOCK_CODE}') {{
        document.getElementById('lockSection').style.display = 'none';
        document.getElementById('controlSection').style.display = 'block';
        checkAll();
      }} else alert('Wrong code!');
    }}

    async function checkStatus(mid) {{
      try {{
        const r = await fetch('/api/status/' + mid);
        const d = await r.json();
        const el = document.getElementById('status-' + mid);
        if(d.online) {{ el.innerHTML = '🟢 ONLINE'; el.className='model-status online'; }}
        else {{ el.innerHTML = '🔴 OFFLINE'; el.className='model-status offline'; }}
      }} catch(e) {{ document.getElementById('status-' + mid).innerHTML = '⚠️ ERROR'; }}
    }}

    function checkAll() {{
      ['miai-v1','miai-v2','miai-v3','miai-v4'].forEach(checkStatus);
    }}

    function genKey() {{
      const mid = document.getElementById('modelSel').value;
      const rand = Array.from({{length:24}}, () => Math.floor(Math.random()*16).toString(16)).join('');
      const key = `miai-live-{LOCK_CODE}-${{rand}}`;
      const box = document.getElementById('keyBox');
      box.style.display = 'block';
      box.innerHTML = `
        <strong>Model:</strong> ${{mid}}<br>
        <strong>Key:</strong> <span style="user-select:all">${{key}}</span><br><br>
        <strong>Endpoint:</strong> https://${{location.host}}/v1/chat/completions<br>
        <strong>Header:</strong> Authorization: Bearer ${{key}}
      `;
    }}

    setInterval(checkAll, 30000);
  </script>
</body>
</html>"""

# ─── Status Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status_all():
    result = {}
    for mid, url in MODEL_URLS.items():
        if not url:
            result[mid] = {"online": False, "reason": "URL not set"}
            continue
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            result[mid] = {"online": r.status_code == 200, "url_set": True}
        except:
            result[mid] = {"online": False, "url_set": True}
    return result

@app.get("/api/status/{model_id}")
async def status_model(model_id: str):
    url = MODEL_URLS.get(model_id)
    if not url:
        return {"online": False, "model": model_id, "reason": "URL not configured"}
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        data = r.json()
        return {"online": True, "model": model_id, **data}
    except:
        return {"online": False, "model": model_id}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "miai-v1", "description": "Qwen2.5-0.5B — Ultra fast, Urdu support"},
            {"id": "miai-v2", "description": "Qwen2.5-1.5B — Balanced speed + quality"},
            {"id": "miai-v3", "description": "SmolLM2-1.7B — Smart multilingual"},
            {"id": "miai-v4", "description": "Phi-2 2.7B — Deep reasoning"},
        ]
    }

# ─── Auth Helper ──────────────────────────────────────────────────────────────────
def validate_key(token: str) -> bool:
    return token.startswith(f"miai-live-{LOCK_CODE}-") or token == "miai-master-token-786"

# ─── Main Chat Completion Gateway ────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Auth check
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token = auth.split(" ", 1)[1]
    if not validate_key(token):
        raise HTTPException(403, "Invalid MI AI API Key")

    body = await request.json()

    # Model selection — default miai-v1
    requested_model = body.get("model", "miai-v1")
    if requested_model not in MODEL_URLS:
        requested_model = "miai-v1"

    target_url = MODEL_URLS.get(requested_model, "")
    if not target_url:
        raise HTTPException(503, f"{requested_model} is currently offline. Check GitHub Actions.")

    body["model"] = requested_model
    endpoint = f"{target_url.rstrip('/')}/v1/chat/completions"

    try:
        resp = requests.post(endpoint, json=body, timeout=90)
        return resp.json()
    except requests.Timeout:
        raise HTTPException(504, f"{requested_model} response timeout. Try a smaller model.")
    except Exception as e:
        raise HTTPException(500, f"Gateway error: {str(e)}")
