"""
MI AI IMAGE SERVER — SDXL-Turbo (text-to-image)
Same ngrok + FastAPI pattern as server.py (text models)
"""

import os
import sys
import time
import threading
import base64
import io
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pyngrok import ngrok
import torch
from diffusers import AutoPipelineForText2Image

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID = "miai-img"
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-img")
PORT = int(os.getenv("PORT", "8001"))
API_KEY = os.getenv("MIAI_API_KEY", "")
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "2"))  # SDXL-Turbo needs only 1-4 steps

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading SDXL-Turbo from {MODEL_PATH} (CPU)...")

try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        local_files_only=True,
        variant=None,
    )
    pipe.to("cpu")
    print("✅ SDXL-Turbo loaded successfully!\n")
except Exception as e:
    print(f"❌ Load failed: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MI AI — Image Generation (SDXL-Turbo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = DEFAULT_STEPS
    guidance_scale: Optional[float] = 0.0  # SDXL-Turbo: guidance_scale=0 by design

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
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/v1/images/generations")
def generate_image(req: ImageRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start = time.time()

        # Cap resolution/steps so CPU doesn't choke — bigger isn't faster here
        width = min(max(req.width or 512, 256), 768)
        height = min(max(req.height or 512, 256), 768)
        steps = min(max(req.steps or DEFAULT_STEPS, 1), 6)

        with torch.no_grad():
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=req.guidance_scale,
                width=width,
                height=height,
            )
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = round(time.time() - start, 2)

        return {
            "id": f"img-{int(time.time())}",
            "object": "image.generation",
            "model": MODEL_ID,
            "latency_seconds": elapsed,
            "data": [
                {"b64_json": b64_img}
            ],
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

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
    print(f"🚀 Starting MI AI image server on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
