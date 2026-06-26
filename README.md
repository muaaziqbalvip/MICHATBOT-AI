# MI AI — 4 Model System v2.0

**MuslimIslam Organization — Own AI Engine**

## Models
| Model | Base | Size | Speed |
|-------|------|------|-------|
| miai-v1 | Qwen2.5-0.5B | ~1GB | ⚡ Ultra Fast |
| miai-v2 | Qwen2.5-1.5B | ~3GB | ⚖️ Balanced |
| miai-v3 | SmolLM2-1.7B | ~3.4GB | 🌍 Multilingual |
| miai-v4 | Phi-2 2.7B | ~5GB | 🧠 Best Quality |

## GitHub Secrets Required
- `NGROK_AUTH_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_PROJECT_NAME`

## Usage
```
POST /v1/chat/completions
Authorization: Bearer miai-live-muaaz19720-XXXX
{"model": "miai-v1", "messages": [{"role":"user","content":"Salaam!"}]}
```
