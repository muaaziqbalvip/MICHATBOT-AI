"""
MI AI FILE SERVER — miai-file
Combined OCR + Audio Processing + General File Processing Engine

Two models loaded together:
  - Qwen2.5-VL-3B-Instruct  → images / scanned docs (OCR, "what's in this
    image", reading handwriting/screenshots, basic document understanding)
  - openai/whisper-small    → audio transcription (voice notes, calls,
    any spoken audio — multilingual, including Urdu/Hinglish)

Single endpoint /v1/files/process accepts a JSON body with an optional
base64 image, optional base64 audio, and/or raw text — all in one call,
matching the "one combined engine" choice for this engine.
"""

import os
import sys
import time
import base64
import io
import threading
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pyngrok import ngrok
import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID = "miai-file"
VL_MODEL_PATH = os.getenv("VL_MODEL_PATH", "./model_files/miai-file-vl")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./model_files/miai-file-whisper")
PORT = int(os.getenv("PORT", "8006"))
API_KEY = os.getenv("MIAI_API_KEY", "")
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

if not NGROK_TOKEN:
    print("❌ NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════

vl_model = None
vl_processor = None
whisper_pipe = None

print(f"\n🔄 Loading Qwen2.5-VL-3B-Instruct (OCR/vision) from {VL_MODEL_PATH}...")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VL_MODEL_PATH,
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    vl_model.eval()
    vl_processor = AutoProcessor.from_pretrained(VL_MODEL_PATH, local_files_only=True)
    print("✅ Vision/OCR model loaded successfully!\n")
except Exception as e:
    print(f"⚠️ Vision/OCR model load failed (OCR endpoints will be disabled): {e}\n")

print(f"🔄 Loading Whisper-small (audio) from {WHISPER_MODEL_PATH}...")
try:
    from transformers import pipeline as hf_pipeline
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_PATH,
        chunk_length_s=30,
        device="cpu",
    )
    print("✅ Whisper audio model loaded successfully!\n")
except Exception as e:
    print(f"⚠️ Whisper model load failed (audio endpoints will be disabled): {e}\n")

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MI AI — miai-file (OCR + Audio + File Processing)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileProcessRequest(BaseModel):
    text: Optional[str] = None              # plain text passthrough/cleanup
    image_b64: Optional[str] = None         # base64 image for OCR/vision
    audio_b64: Optional[str] = None         # base64 audio for transcription
    audio_format: Optional[str] = "wav"     # wav/mp3/m4a/ogg
    instruction: Optional[str] = "Extract and describe all text and content from this image."
    language_hint: Optional[str] = None     # e.g. "ur" for Urdu, helps Whisper

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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def run_ocr(image_b64: str, instruction: str) -> str:
    if vl_model is None or vl_processor is None:
        raise HTTPException(status_code=503, detail="OCR/vision model not loaded on this engine")
    try:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    text_prompt = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vl_processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

    with torch.no_grad():
        generated_ids = vl_model.generate(**inputs, max_new_tokens=400)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = vl_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output[0].strip() if output else ""

def run_transcription(audio_b64: str, audio_format: str, language_hint: Optional[str]) -> str:
    if whisper_pipe is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded on this engine")
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    suffix = f".{audio_format or 'wav'}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        kwargs = {}
        if language_hint:
            kwargs["generate_kwargs"] = {"language": language_hint}
        result = whisper_pipe(tmp.name, **kwargs)
    return result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "purpose": "OCR + audio + general file processing",
        "ocr_available": vl_model is not None,
        "audio_available": whisper_pipe is not None,
    }

@app.post("/v1/files/process")
def process_file(req: FileProcessRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)

    if not req.text and not req.image_b64 and not req.audio_b64:
        raise HTTPException(status_code=400, detail="Provide at least one of: text, image_b64, audio_b64")

    start = time.time()
    result = {
        "id": f"file-{int(time.time())}",
        "object": "file.process",
        "model": MODEL_ID,
    }

    try:
        if req.image_b64:
            result["ocr_text"] = run_ocr(req.image_b64, req.instruction or "Extract all text from this image.")

        if req.audio_b64:
            result["transcription"] = run_transcription(req.audio_b64, req.audio_format, req.language_hint)

        if req.text:
            # Plain text passthrough — kept simple/cheap, no LLM call needed here.
            result["text_echo"] = req.text.strip()

        result["latency_seconds"] = round(time.time() - start, 2)
        return result

    except HTTPException:
        raise
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
    print(f"🚀 Starting miai-file server on port {PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
