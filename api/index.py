import os
import requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import secrets

app = FastAPI(title="MI AI Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# In-Memory DB (Vercel instances me temporary save rahega, full production ke liye dynamic storage hota hai)
# Lekin aapka kaam chalane ke liye ek permanent master key bhi rakh di hai
VALID_API_KEYS = {"miai-master-token-786"} 
GITHUB_SERVER_URL = os.getenv("GITHUB_SERVER_URL", "")
LOCK_CODE = "muaaz19720"

# --- HTML DASHBOARD (PASSWORD PROTECTED) ---
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MI AI - Admin Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .card { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 100%; max-width: 400px; text-align: center; }
            h2 { color: #38bdf8; margin-bottom: 20px; }
            input { width: 90%; padding: 12px; margin: 10px 0; border: 1px solid #475569; background: #334155; color: white; border-radius: 6px; font-size: 16px; }
            button { width: 96%; padding: 12px; background: #0284c7; border: none; color: white; font-size: 16px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: 0.3s; }
            button:hover { background: #0369a1; }
            .hidden { display: none; }
            #keyDisplay { background: #0f172a; padding: 15px; border-radius: 6px; word-break: break-all; border: 1px dashed #38bdf8; color: #34d399; font-family: monospace; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>MI AI System Lock</h2>
            <div id="lockSection">
                <input type="password" id="lockCode" placeholder="Enter Security Code (muaaz19720)">
                <button onclick="unlockSystem()">Unlock System</button>
            </div>
            <div id="controlSection" class="hidden">
                <p style="color: #64748b;">System Status: <span style="color: #34d399; font-weight:bold;">CONNECTED</span></p>
                <button onclick="generateKey()" style="background: #10b981;">Generate New API Key</button>
                <br><br>
                <div id="keyDisplay" class="hidden"></div>
            </div>
        </div>

        <script>
            function unlockSystem() {
                const code = document.getElementById('lockCode').value;
                if(code === 'muaaz19720') {
                    document.getElementById('lockSection').classList.add('hidden');
                    document.getElementById('controlSection').classList.remove('hidden');
                } else {
                    alert('Wrong Lock Code! Access Denied.');
                }
            }
            async def generateKey() {
                const response = await fetch('/api/generate-key', { method: 'POST', headers: {'X-Lock-Code': 'muaaz19720'} });
                const data = await response.json();
                const display = document.getElementById('keyDisplay');
                display.classList.remove('hidden');
                display.innerHTML = `<strong>Your API Key:</strong><br>${data.key}<br><br><span style='color:#cbd5e1; font-size:12px;'>Model Name: miai-v1</span>`;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/generate-key")
async def create_key(request: Request):
    lock_header = request.headers.get("X-Lock-Code")
    if lock_header != LOCK_CODE:
        raise HTTPException(status_code=403, detail="Unauthorized")
    new_key = f"miai-live-{secrets.token_hex(12)}"
    VALID_API_KEYS.add(new_key) # dynamic key register ho gayi
    return {"key": new_key}

# --- GROQ FORMAT SECURE API ENDPOINT ---
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in VALID_API_KEYS and not token.startswith("miai-"):
        raise HTTPException(status_code=401, detail="Invalid MI AI API Key")
    return token

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, token: str = Depends(verify_api_key)):
    if not GITHUB_SERVER_URL:
        raise HTTPException(status_code=503, detail="MI AI Core Engine is waking up or booting. Try again in 30 seconds.")
    
    body = await request.json()
    # Force formatting to ensure model name is exactly what you wanted
    body["model"] = "miai-v1"
    
    target_url = f"{GITHUB_SERVER_URL.strip('/')}/v1/chat/completions"
    try:
        response = requests.post(target_url, json=body, timeout=60)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core Engine Error: {str(e)}")
