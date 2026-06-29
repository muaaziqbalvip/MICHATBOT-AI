"""
MI AI VIDEO SERVER — text-to-video-ms-1.7b (CPU)
Same ngrok + FastAPI pattern as server.py (text models)

⚠️ HONEST NOTE: This runs on CPU only (no GPU on free GitHub Actions runners).
A single short video (16-24 frames) can take 5-15 minutes to generate.
This is a hardware limit, not a bug — there is no way to make raw diffusion
video generation "instant" without a GPU.
"""

import os
import sys
import time
import threading
import base64
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pyngrok import ngrok
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID = "miai-video"
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-video")
PORT = int(os.getenv("PORT", "8002"))
API_KEY = os.getenv("MIAI_API_KEY", "")
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
DEFAULT_FRAMES = int(os.getenv("DEFAULT_FRAMES", "16"))  # 16 frames @ 8fps = 2s
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

OUTPUT_DIR = "./video_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Loading text-to-video-ms-1.7b from {MODEL_PATH} (CPU — this is slow)...")

try:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        local_files_only=True,
        variant=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")
    print("✅ Video model loaded successfully!\n")
except Exception as e:
    print(f"❌ Load failed: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MI AI — Video Generation (text-to-video-ms-1.7b)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    prompt: str
    num_frames: Optional[int] = DEFAULT_FRAMES
    steps: Optional[int] = DEFAULT_STEPS

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

@app.post("/v1/videos/generations")
def generate_video(req: VideoRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    try:
        start = time.time()

        # Keep frames/steps capped — CPU video gen is slow, this avoids timeouts
        num_frames = min(max(req.num_frames or DEFAULT_FRAMES, 8), 24)
        steps = min(max(req.steps or DEFAULT_STEPS, 10), 30)

        print(f"⏳ Generating video: '{req.prompt[:60]}' ({num_frames} frames, {steps} steps)...")

        with torch.no_grad():
            result = pipe(
                req.prompt,
                num_inference_steps=steps,
                num_frames=num_frames,
            )
        frames = result.frames[0]

        out_path = os.path.join(OUTPUT_DIR, f"video_{int(time.time())}.mp4")
        export_to_video(frames, out_path)

        with open(out_path, "rb") as f:
            b64_video = base64.b64encode(f.read()).decode("utf-8")

        try:
            os.remove(out_path)
        except OSError:
            pass

        elapsed = round(time.time() - start, 2)
        print(f"✅ Video done in {elapsed}s")

        return {
            "id": f"vid-{int(time.time())}",
            "object": "video.generation",
            "model": MODEL_ID,
            "latency_seconds": elapsed,
            "data": [
                {"b64_mp4": b64_video}
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
    print(f"🚀 Starting MI AI video server on port {PORT}...\n")
    # Higher timeout-friendly server since requests can take minutes
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", timeout_keep_alive=900)
