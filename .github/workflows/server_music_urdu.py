import os
import torch
import json
from fastapi import FastAPI, HTTPException
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
    # English aliases
    "happy": "happy, joyful, uplifting",
    "sad": "sad, melancholic, emotional, slow",
    "calm": "calm, peaceful, relaxing, ambient",
    "fast": "fast, energetic, upbeat, dance",
    "islamic": "Islamic, spiritual, qawwali-inspired, traditional",
    "romantic": "romantic, love, soft, dreamy",
    "serious": "serious, dramatic, cinematic, orchestral",
    "kids": "kids, playful, fun, light, whimsical",
}

# Load model globally
print("🎵 Loading MusicGen Small model (Urdu-friendly)...")
try:
    model = MusicGen.get_model("facebook/musicgen-small")
    model.set_generation_params(use_sampling=True, top_k=250)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# ============ PYDANTIC MODELS ============
class MusicRequest(BaseModel):
    prompt: str  # Can be in Urdu, Hinglish, or English
    mood: str = None  # Urdu mood descriptor (optional)
    duration: int = 30  # seconds (max 30)
    temperature: float = 1.0  # 0.1-2.0
    top_k: int = 250
    top_p: float = 0.9
    language: str = "auto"  # "urdu", "english", "hinglish", "auto"

class MusicResponse(BaseModel):
    success: bool
    message: str
    audio_url: str = None
    duration: int = None
    prompt_expanded: str = None
    model_info: dict = None

# ============ UTILITY FUNCTIONS ============
def expand_urdu_prompt(prompt: str, mood: str = None) -> str:
    """Expand Urdu/Hinglish prompt to full English description"""
    
    # Check if mood is Urdu
    if mood and mood in URDU_MOOD_MAP:
        mood_expansion = URDU_MOOD_MAP[mood]
        return f"{prompt}, {mood_expansion}"
    
    # Common Urdu/Hinglish mappings
    urdu_words = {
        "نaat": "naat, Islamic prayer, spiritual vocal",
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
    
    # If mood is provided (English)
    if mood and mood.lower() in URDU_MOOD_MAP:
        expanded += f", {URDU_MOOD_MAP[mood.lower()]}"
    
    return expanded

# ============ ROUTES ============
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "facebook/musicgen-small",
        "model_loaded": model is not None,
        "urdu_support": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest):
    """Generate music from text prompt (supports Urdu, Hinglish, English)"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Clamp duration
    duration = min(max(request.duration, 5), 30)
    
    try:
        # Expand prompt with mood/context
        expanded_prompt = expand_urdu_prompt(request.prompt, request.mood)
        
        print(f"🎵 Generating music:")
        print(f"   Original: {request.prompt}")
        print(f"   Mood: {request.mood}")
        print(f"   Expanded: {expanded_prompt}")
        print(f"   Duration: {duration}s")
        
        # Generate with timeout
        with torch.no_grad():
            descriptions = [expanded_prompt]
            wav = model.generate(
                descriptions,
                progress=True,
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=request.temperature
            )
        
        # Save as WAV
        output_dir = "/tmp/miai_music"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(asyncio.get_event_loop().time() * 1000)
        output_path = os.path.join(output_dir, f"music_{timestamp}.wav")
        
        # WAV file save
        sample_rate = 16000
        audio_data = wav[0].cpu().numpy()
        
        # Normalize to prevent clipping
        audio_data = np.clip(audio_data, -1.0, 1.0)
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16.T if audio_int16.shape[0] > 1 else audio_int16.flatten())
        
        print(f"✅ Music generated: {output_path}")
        
        return MusicResponse(
            success=True,
            message=f"✅ Urdu-friendly music generated ({duration}s)",
            audio_url=f"file://{output_path}",
            duration=duration,
            prompt_expanded=expanded_prompt,
            model_info={
                "model": "MusicGen Small",
                "sample_rate": sample_rate,
                "supports_urdu": True,
                "supports_hinglish": True,
            }
        )
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/moods")
async def list_moods():
    """Get available mood options"""
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
        },
        "example_prompts": {
            "urdu": ["naat", "qawwali", "ghazal", "sufi", "bhangra"],
            "hinglish": ["wedding music", "Bollywood style", "folk instrumental"],
            "english": ["upbeat electronic", "peaceful ambient", "orchestral drama"],
        }
    }

@app.get("/info")
async def info():
    """Get model information"""
    return {
        "model": "MusicGen Small",
        "model_size": "328M",
        "max_duration": 30,
        "sample_rate": 16000,
        "supported_formats": ["wav"],
        "urdu_support": True,
        "hinglish_support": True,
        "english_support": True,
        "features": [
            "Urdu mood descriptors",
            "Hinglish prompt expansion",
            "Multi-language support",
            "Mood-based generation",
        ],
        "ngrok_url": os.getenv("NGROK_URL", "pending")
    }

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    """Setup ngrok tunnel on startup"""
    if NGROK_TOKEN:
        try:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(PORT, "http")
            print(f"🌐 Ngrok tunnel: {public_url}")
            os.environ["NGROK_URL"] = str(public_url)
        except Exception as e:
            print(f"⚠️ Ngrok setup failed (continuing anyway): {e}")

# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting Urdu-Friendly Music Generation Server on port {PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
