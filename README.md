# MI AI — Multi-Model AI System v3.1

**MuslimIslam Organization — Own AI Engine**

## Core Chat Models
| Model | Base | Size | Speed |
|-------|------|------|-------|
| miai-v1 | Qwen2.5-0.5B | ~1GB | ⚡ Ultra Fast |
| miai-v2 | Qwen2.5-1.5B | ~3GB | ⚖️ Balanced |
| miai-v3 | SmolLM2-1.7B | ~3.4GB | 🌍 Multilingual |
| miai-v4 | Phi-2 2.7B | ~5GB | 🧠 Best Quality |

## Specialist Text Models
| Model | Base | Purpose |
|-------|------|---------|
| miai-call | Qwen2.5-1.5B-Instruct | Fast + balanced AI Call assistant (short, spoken-style replies) |
| miai-urdu | Qwen2.5-1.5B-Instruct (Urdu-primed) | Dedicated Urdu text — always replies in Urdu script |
| miai-agent | Qwen2.5-3B-Instruct | Online task / tool-use agent (free DuckDuckGo web search) |
| miai-file | Qwen2.5-VL-3B-Instruct + Whisper-small | Combined OCR + audio transcription + file processing |

## Media Engines
| Model | Base | Purpose |
|-------|------|---------|
| miai-img | SDXL-Turbo | Image generation |
| miai-video | text-to-video-ms-1.7b | Video generation (async job, CPU, 5-15 min) |
| miai-music | MusicGen Small | Music generation (Urdu/Hinglish/English prompts) |

## GitHub Secrets Required
- `NGROK_AUTH_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_PROJECT_NAME`
- `VERCEL_DEPLOY_HOOK_URL` (optional but recommended — auto-redeploys Vercel after each engine restart so it picks up the fresh ngrok URL)

## Usage

**Chat / specialist text models** (same endpoint for all of: miai-v1..v4, miai-call, miai-urdu, miai-agent):
```
POST /v1/chat/completions
Authorization: Bearer miai-live-muaaz19720-XXXX
{"model": "miai-call", "messages": [{"role":"user","content":"Salaam!"}]}
```

**Online task agent (dedicated endpoint, with web search + sources):**
```
POST /v1/agent/tasks
Authorization: Bearer miai-live-muaaz19720-XXXX
{"task": "Aaj ka USD to PKR rate kya hai?"}
```

**File / OCR / audio processing:**
```
POST /v1/files/process
Authorization: Bearer miai-live-muaaz19720-XXXX
{"image_b64": "...", "audio_b64": "...", "audio_format": "mp3", "instruction": "Extract all text"}
```

**Music generation:**
```
POST /v1/audio/music
Authorization: Bearer miai-live-muaaz19720-XXXX
{"prompt": "Ek pur-sukoon qawwali jaisi dhun", "duration": 15}
```

**Image / video generation:** unchanged — `/v1/images/generations`, `/v1/videos/generations` (async, poll `/v1/videos/status/{job_id}`).

## Notes
- Each engine runs as its own GitHub Actions workflow (`.github/workflows/miai_*.yml`), auto-restarting every ~6 hours and syncing its live ngrok URL into the matching Vercel env var.
- `miai_music.yml` was previously broken (pointed at a missing `server_music.py` and an inconsistent model name) — now fixed to run `server_music_urdu.py` with `facebook/musicgen-small` consistently end-to-end.
- `miai-test-console.html` now has tabs for Music, Agent Task, and File/OCR/Audio in addition to the original Text/Image/Video tabs.

