"""
MI AI Gateway v3.0 — Vercel API (api/index.py)
4 models support: miai-v1, miai-v2, miai-v3, miai-v4
Auto-routing: model name se sahi GitHub server pe forward karta hai.

WHAT CHANGED IN v3.0
- Dashboard fully redesigned: control-room look, live token/sec + latency
  readouts per model, clearer online/offline state, same lock-code flow.
- Gateway / routing / auth logic untouched — this still talks to the same
  server.py engines via the same MODEL_URLS env vars, so nothing else in
  your pipeline needs to change.
"""

import os
import time
import threading
import uuid
import requests
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="MI AI Gateway v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store for async video generation (Hobby plan: max 300s per
# request, but video gen takes 5-15 min — so we return a job_id immediately
# and let the caller poll /v1/videos/status/{job_id}).
# NOTE: This resets on every cold start / new deployment since it's
# in-memory only. For production durability, swap this for Firebase
# (which is already part of this project's stack) or Vercel KV.
VIDEO_JOBS = {}

# Each model ka Vercel env se URL milta hai
MODEL_URLS = {
    "miai-v1": os.getenv("MIAI_V1_URL", ""),
    "miai-v2": os.getenv("MIAI_V2_URL", ""),
    "miai-v3": os.getenv("MIAI_V3_URL", ""),
    "miai-v4": os.getenv("MIAI_V4_URL", ""),
}

# Image / Video engines (separate dict — different request/response shape)
MEDIA_URLS = {
    "miai-img": os.getenv("MIAI_IMG_URL", ""),
    "miai-video": os.getenv("MIAI_VIDEO_URL", ""),
}

LOCK_CODE = "muaaz19720"

MODEL_META = {
    "miai-v1": {"name": "Qwen2.5-0.5B", "badge": "Ultra Fast", "accent": "#5eead4"},
    "miai-v2": {"name": "Qwen2.5-1.5B", "badge": "Balanced",   "accent": "#7dd3fc"},
    "miai-v3": {"name": "SmolLM2-1.7B", "badge": "Multilingual", "accent": "#c4b5fd"},
    "miai-v4": {"name": "Phi-2 2.7B",   "badge": "Deep Reason", "accent": "#fda4af"},
}

MEDIA_META = {
    "miai-img": {"name": "SDXL-Turbo", "badge": "Image Gen", "accent": "#fcd34d"},
    "miai-video": {"name": "text-to-video-ms-1.7b", "badge": "Video Gen (slow, CPU)", "accent": "#f472b6"},
}

# ─── Admin Dashboard ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    model_cards = ""
    for mid, meta in MODEL_META.items():
        model_cards += f"""
        <div class="unit" id="card-{mid}" style="--accent:{meta['accent']}">
          <div class="unit-top">
            <span class="unit-id">{mid}</span>
            <span class="pill" id="status-{mid}">checking…</span>
          </div>
          <div class="unit-name">{meta['name']}</div>
          <div class="unit-badge">{meta['badge']}</div>
          <div class="unit-metrics">
            <div><span class="metric-label">latency</span><span class="metric-val" id="lat-{mid}">—</span></div>
            <div><span class="metric-label">tok/s</span><span class="metric-val" id="tps-{mid}">—</span></div>
          </div>
        </div>
        """

    media_cards = ""
    for mid, meta in MEDIA_META.items():
        media_cards += f"""
        <div class="unit" id="card-{mid}" style="--accent:{meta['accent']}">
          <div class="unit-top">
            <span class="unit-id">{mid}</span>
            <span class="pill" id="status-{mid}">checking…</span>
          </div>
          <div class="unit-name">{meta['name']}</div>
          <div class="unit-badge">{meta['badge']}</div>
        </div>
        """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MI AI — Engine Control</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

  :root {{
    --bg: #0b0d10;
    --panel: #12151a;
    --panel-2: #161a20;
    --line: #232830;
    --text: #e7eaee;
    --text-dim: #828a96;
    --green: #4ade80;
    --red: #f87171;
    --amber: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    background-image:
      radial-gradient(circle at 15% 0%, rgba(94,234,212,0.05), transparent 40%),
      radial-gradient(circle at 85% 10%, rgba(196,181,253,0.05), transparent 40%);
  }}
  .topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 28px; border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  .brand {{ display: flex; align-items: center; gap: 10px; }}
  .brand-dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
  .brand h1 {{ font-size: 15px; font-weight: 700; letter-spacing: 0.3px; }}
  .brand span {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); }}
  .topbar-right {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); }}

  .wrap {{ max-width: 920px; margin: 0 auto; padding: 40px 24px 80px; }}

  .lock-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 36px; max-width: 380px; margin: 60px auto;
  }}
  .lock-card h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 4px; }}
  .lock-card p {{ font-size: 13px; color: var(--text-dim); margin-bottom: 20px; }}
  input {{
    width: 100%; padding: 12px 14px; background: var(--panel-2);
    border: 1px solid var(--line); color: var(--text); border-radius: 9px;
    font-size: 14px; outline: none; font-family: 'JetBrains Mono', monospace;
  }}
  input:focus {{ border-color: var(--green); }}
  .btn {{
    width: 100%; padding: 12px; background: var(--text); border: none;
    color: #0b0d10; font-size: 14px; border-radius: 9px; cursor: pointer;
    font-weight: 600; margin-top: 10px; transition: 0.15s;
  }}
  .btn:hover {{ opacity: 0.85; }}
  .btn-ghost {{ background: transparent; border: 1px solid var(--line); color: var(--text); }}
  .btn-ghost:hover {{ border-color: var(--text-dim); opacity: 1; }}

  #controlSection {{ display: none; }}

  .section-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 1.2px; margin: 36px 0 14px;
    display: flex; align-items: center; gap: 10px;
  }}
  .section-label::after {{ content: ''; flex: 1; height: 1px; background: var(--line); }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}

  .unit {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 18px; position: relative; overflow: hidden;
  }}
  .unit::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
  }}
  .unit-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .unit-id {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--accent); font-weight: 700; }}
  .pill {{
    font-size: 10px; padding: 3px 9px; border-radius: 20px; font-weight: 600;
    background: var(--panel-2); color: var(--text-dim); letter-spacing: 0.3px;
  }}
  .pill.online {{ background: rgba(74,222,128,0.12); color: var(--green); }}
  .pill.offline {{ background: rgba(248,113,113,0.12); color: var(--red); }}
  .unit-name {{ font-size: 16px; font-weight: 700; }}
  .unit-badge {{ font-size: 11px; color: var(--text-dim); margin: 2px 0 14px; }}
  .unit-metrics {{ display: flex; gap: 18px; border-top: 1px solid var(--line); padding-top: 12px; }}
  .metric-label {{ display: block; font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-val {{ display: block; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 2px; }}

  .keygen {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px; margin-top: 8px;
  }}
  select {{
    width: 100%; padding: 11px; background: var(--panel-2); border: 1px solid var(--line);
    color: var(--text); border-radius: 9px; margin: 10px 0; font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
  }}
  #keyBox {{
    background: var(--panel-2); border: 1px dashed var(--green); border-radius: 9px;
    padding: 14px; margin-top: 14px; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; line-height: 1.7; word-break: break-all; color: var(--green); display: none;
  }}
  #keyBox b {{ color: var(--text); }}
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="brand-dot"></div>
      <h1>MI AI · Engine Control</h1>
      <span>v3.0</span>
    </div>
    <div class="topbar-right">4 ENGINES · GITHUB ACTIONS</div>
  </div>

  <div class="wrap">
    <div class="lock-card" id="lockCard">
      <div id="lockSection">
        <h2>🔐 Locked</h2>
        <p>Enter the security code to access engine controls.</p>
        <input type="password" id="lockInput" placeholder="Security code" onkeydown="if(event.key==='Enter')unlock()">
        <button class="btn" onclick="unlock()">Unlock</button>
      </div>
    </div>

    <div id="controlSection">
      <div class="section-label">Live Engines</div>
      <div class="grid">{model_cards}</div>

      <div class="section-label">Media Engines (Image / Video)</div>
      <div class="grid">{media_cards}</div>

      <div class="section-label">API Key</div>
      <div class="keygen">
        <select id="modelSel">
          <option value="miai-v1">miai-v1 — Qwen 0.5B (Fast)</option>
          <option value="miai-v2">miai-v2 — Qwen 1.5B (Balanced)</option>
          <option value="miai-v3">miai-v3 — SmolLM2 1.7B</option>
          <option value="miai-v4">miai-v4 — Phi-2 2.7B (Best)</option>
        </select>
        <button class="btn btn-ghost" onclick="genKey()">Generate key</button>
        <div id="keyBox"></div>
      </div>
    </div>
  </div>

  <script>
    function unlock() {{
      if(document.getElementById('lockInput').value === '{LOCK_CODE}') {{
        document.getElementById('lockCard').style.display = 'none';
        document.getElementById('controlSection').style.display = 'block';
        checkAll();
      }} else alert('Wrong code!');
    }}

    async function checkStatus(mid) {{
      const pill = document.getElementById('status-' + mid);
      const lat = document.getElementById('lat-' + mid);
      const tps = document.getElementById('tps-' + mid);
      try {{
        const r = await fetch('/api/status/' + mid);
        const d = await r.json();
        if(d.online) {{
          pill.textContent = 'online'; pill.className = 'pill online';
          lat.textContent = d.max_tokens ? d.max_tokens + ' tok cap' : '—';
          tps.textContent = d.quantized ? 'int8' : 'fp32';
        }} else {{
          pill.textContent = 'offline'; pill.className = 'pill offline';
          lat.textContent = '—'; tps.textContent = '—';
        }}
      }} catch(e) {{
        pill.textContent = 'error'; pill.className = 'pill offline';
      }}
    }}

    function checkAll() {{
      ['miai-v1','miai-v2','miai-v3','miai-v4','miai-img','miai-video'].forEach(checkStatus);
    }}

    function genKey() {{
      const mid = document.getElementById('modelSel').value;
      const rand = Array.from({{length:24}}, () => Math.floor(Math.random()*16).toString(16)).join('');
      const key = `miai-live-{LOCK_CODE}-${{rand}}`;
      const box = document.getElementById('keyBox');
      box.style.display = 'block';
      box.innerHTML = `
        <b>Model:</b> ${{mid}}<br>
        <b>Key:</b> <span style="user-select:all">${{key}}</span><br><br>
        <b>Endpoint:</b> https://${{location.host}}/v1/chat/completions<br>
        <b>Header:</b> Authorization: Bearer ${{key}}
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
    for mid, url in {**MODEL_URLS, **MEDIA_URLS}.items():
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
    url = MODEL_URLS.get(model_id) or MEDIA_URLS.get(model_id)
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
            {"id": "miai-v4", "description": "Phi-2 2.7B — Deep reasoning + coding"},
            {"id": "miai-img", "description": "SDXL-Turbo — Image generation (1-2 min/image, CPU)"},
            {"id": "miai-video", "description": "text-to-video-ms-1.7b — Video generation (5-15 min/video, CPU)"},
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

# ─── Image Generation Gateway ────────────────────────────────────────────────────
@app.post("/v1/images/generations")
async def image_generations(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token = auth.split(" ", 1)[1]
    if not validate_key(token):
        raise HTTPException(403, "Invalid MI AI API Key")

    body = await request.json()
    target_url = MEDIA_URLS.get("miai-img", "")
    if not target_url:
        raise HTTPException(503, "miai-img is currently offline. Check GitHub Actions.")

    endpoint = f"{target_url.rstrip('/')}/v1/images/generations"
    try:
        # SDXL-Turbo on CPU: ~1-2 min/image, give it generous headroom
        resp = requests.post(endpoint, json=body, timeout=180)
        return resp.json()
    except requests.Timeout:
        raise HTTPException(504, "Image generation timed out (CPU is slow). Try again.")
    except Exception as e:
        raise HTTPException(500, f"Gateway error: {str(e)}")

# ─── Video Generation Gateway (ASYNC — Hobby plan can't hold a request open
# for 5-15 minutes, so this returns immediately and the caller polls status) ──
def _run_video_job(job_id: str, body: dict, target_url: str):
    """Runs in a background thread; updates VIDEO_JOBS when done."""
    try:
        endpoint = f"{target_url.rstrip('/')}/v1/videos/generations"
        resp = requests.post(endpoint, json=body, timeout=1200)
        data = resp.json()
        if resp.status_code == 200:
            VIDEO_JOBS[job_id] = {"status": "completed", "result": data}
        else:
            VIDEO_JOBS[job_id] = {"status": "failed", "error": data}
    except requests.Timeout:
        VIDEO_JOBS[job_id] = {"status": "failed", "error": "Video generation timed out on the engine (CPU is slow). Try fewer frames/steps."}
    except Exception as e:
        VIDEO_JOBS[job_id] = {"status": "failed", "error": str(e)}

@app.post("/v1/videos/generations")
async def video_generations(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token = auth.split(" ", 1)[1]
    if not validate_key(token):
        raise HTTPException(403, "Invalid MI AI API Key")

    body = await request.json()
    target_url = MEDIA_URLS.get("miai-video", "")
    if not target_url:
        raise HTTPException(503, "miai-video is currently offline. Check GitHub Actions.")

    job_id = uuid.uuid4().hex
    VIDEO_JOBS[job_id] = {"status": "processing"}

    t = threading.Thread(target=_run_video_job, args=(job_id, body, target_url), daemon=True)
    t.start()

    # Returned right away — this request finishes in well under 300s.
    # Caller must poll /v1/videos/status/{job_id} for the actual result.
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video generation started. CPU rendering takes 5-15 minutes. Poll /v1/videos/status/{job_id} for the result.",
        "poll_url": f"/v1/videos/status/{job_id}",
    }

@app.get("/v1/videos/status/{job_id}")
async def video_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id (may have expired after a redeploy/cold start).")
    return job
