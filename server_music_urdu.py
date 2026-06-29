import os
import torch
import json
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from audiocraft.models import MusicGen
from scipy.io import wavfile
import numpy as np
from pyngrok import ngrok
import asyncio
from datetime import datetime

# ============ SETUP ============
app = FastAPI(title="MIAI Music Generator - Urdu Friendly")
MODEL_PATH = os.getenv("MODEL_PATH", "./model_files/miai-music")
PORT = int(os.getenv("PORT", 8002))
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

# Urdu mood mappings
URDU_MOOD_MAP = {
    "خوشی": "happy, joyful, uplifting",
    "غمی": "sad, melancholic, emotional, slow",
    "آرام": "calm, peaceful, relaxing, ambient",
    "تیز": "fast, energetic, upbeat, dance",
    "اسلامی": "Islamic, spiritual, qawwali-inspired, traditional",
    "رومانوی": "romantic, love, soft, dreamy",
    "سنجیدہ": "serious, dramatic, cinematic, orchestral",
    "بچوں": "kids, playful, fun, light, whimsical",
    "happy": "happy, joyful, uplifting",
    "sad": "sad, melancholic, emotional, slow",
    "calm": "calm, peaceful, relaxing, ambient",
    "fast": "fast, energetic, upbeat, dance",
    "islamic": "Islamic, spiritual, qawwali-inspired, traditional",
    "romantic": "romantic, love, soft, dreamy",
    "serious": "serious, dramatic, cinematic, orchestral",
    "kids": "kids, playful, fun, light, whimsical",
}

# Load model globally — LOCAL PATH use karo
print(f"🎵 Loading MusicGen model from: {MODEL_PATH}")
try:
    # Pehle local path try karo, fallback to HF download
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"✅ Using local model from {MODEL_PATH}")
        model = MusicGen.get_model(MODEL_PATH)
    else:
        print("⚠️ Local model not found, downloading from HuggingFace...")
        model = MusicGen.get_model("facebook/musicgen-large")
    model.set_generation_params(use_sampling=True, top_k=250)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# ============ PYDANTIC MODELS ============
class MusicRequest(BaseModel):
    prompt: str
    mood: str = None
    duration: int = 30
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.9
    language: str = "auto"

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_url: str = None
    audio_base64: str = None
    duration: int = None
    prompt_expanded: str = None
    model_info: dict = None

# ============ UTILITY FUNCTIONS ============
def expand_urdu_prompt(prompt: str, mood: str = None) -> str:
    if mood and mood in URDU_MOOD_MAP:
        mood_expansion = URDU_MOOD_MAP[mood]
        return f"{prompt}, {mood_expansion}"
    
    urdu_words = {
        "naat": "naat, Islamic prayer, spiritual vocal",
        "qawwali": "qawwali, Sufi music, devotional, rhythmic",
        "ghazal": "ghazal, classical Indian, poetic, vocal",
        "wedding": "celebration, festive, drums, energetic",
        "sufi": "Sufi music, spiritual, meditative, oud, traditional",
        "bollywood": "Bollywood, Indian cinema, orchestral, dramatic",
        "classical": "classical Indian, sitar, tabla, melodic",
        "folk": "folk music, acoustic, traditional, instrumental",
        "bhangra": "Bhangra, Punjabi, drums, festive, upbeat",
    }
    
    expanded = prompt
    for key, val in urdu_words.items():
        if key.lower() in prompt.lower():
            expanded = expanded.replace(key, val)
    
    if mood and mood.lower() in URDU_MOOD_MAP:
        expanded += f", {URDU_MOOD_MAP[mood.lower()]}"
    
    return expanded

# ============ ROUTES ============
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "facebook/musicgen-large",
        "model_loaded": model is not None,
        "urdu_support": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    duration = min(max(request.duration, 5), 30)
    
    try:
        expanded_prompt = expand_urdu_prompt(request.prompt, request.mood)
        
        print(f"🎵 Generating music:")
        print(f"   Original: {request.prompt}")
        print(f"   Mood: {request.mood}")
        print(f"   Expanded: {expanded_prompt}")
        print(f"   Duration: {duration}s")
        
        model.set_generation_params(
            use_sampling=True,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            duration=duration
        )
        
        with torch.no_grad():
            wav = model.generate([expanded_prompt], progress=True)
        
        # Save WAV
        output_dir = "/tmp/miai_music"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(datetime.now().timestamp() * 1000)
        filename = f"music_{timestamp}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Get sample rate from model
        sample_rate = model.sample_rate
        audio_data = wav[0].cpu().numpy()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(
            output_path,
            sample_rate,
            audio_int16.T if audio_int16.shape[0] > 1 else audio_int16.flatten()
        )
        
        # Base64 encode karo taake browser me direct play ho
        with open(output_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        print(f"✅ Music generated: {output_path}")
        
        # Ngrok URL se serve karo
        ngrok_url = os.getenv("NGROK_URL", "")
        if ngrok_url:
            audio_url = f"{ngrok_url}/audio/{filename}"
        else:
            audio_url = f"http://127.0.0.1:{PORT}/audio/{filename}"
        
        return MusicResponse(
            success=True,
            message=f"✅ Music generated ({duration}s)",
            audio_url=audio_url,
            audio_base64=f"data:audio/wav;base64,{audio_b64}",
            duration=duration,
            prompt_expanded=expanded_prompt,
            model_info={
                "model": "MusicGen Large",
                "sample_rate": sample_rate,
                "supports_urdu": True,
                "supports_hinglish": True,
            }
        )
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Audio file serve karo"""
    path = f"/tmp/miai_music/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")

@app.get("/moods")
async def list_moods():
    return {
        "urdu_moods": {
            "خوشی": "Happy/Joyful",
            "غمی": "Sad/Melancholic",
            "آرام": "Calm/Peaceful",
            "تیز": "Fast/Energetic",
            "اسلامی": "Islamic/Spiritual",
            "رومانوی": "Romantic",
            "سنجیدہ": "Serious/Dramatic",
            "بچوں": "Kids/Playful",
        },
        "english_moods": {
            "happy": "Happy/Joyful",
            "sad": "Sad/Melancholic",
            "calm": "Calm/Peaceful",
            "fast": "Fast/Energetic",
            "islamic": "Islamic/Spiritual",
            "romantic": "Romantic",
            "serious": "Serious/Dramatic",
            "kids": "Kids/Playful",
        }
    }

@app.get("/info")
async def info():
    return {
        "model": "MusicGen Large",
        "model_size": "3.3GB",
        "max_duration": 30,
        "sample_rate": getattr(model, 'sample_rate', 32000) if model else 32000,
        "supported_formats": ["wav"],
        "urdu_support": True,
        "hinglish_support": True,
        "english_support": True,
        "ngrok_url": os.getenv("NGROK_URL", "pending")
    }

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    if NGROK_TOKEN:
        try:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(PORT, "http")
            url_str = str(public_url)
            print(f"🌐 Ngrok tunnel: {url_str}")
            os.environ["NGROK_URL"] = url_str
        except Exception as e:
            print(f"⚠️ Ngrok setup failed: {e}")

# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting Urdu-Friendly Music Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
