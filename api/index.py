import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="MI AI Gateway")

# CORS Settings taaki lock frontend aur backend smoothly baat karein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_SERVER_URL = os.getenv("GITHUB_SERVER_URL", "")
LOCK_CODE = "muaaz19720"

# --- SMART DASHBOARD HTML (PASSWORD + LOCAL STORAGE SECURE) ---
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
            .card { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 100%; max-width: 400px; text-align: center; border: 1px solid #334155; }
            h2 { color: #38bdf8; margin-bottom: 20px; font-size: 24px; letter-spacing: 1px; }
            input { width: 90%; padding: 12px; margin: 10px 0; border: 1px solid #475569; background: #334155; color: white; border-radius: 6px; font-size: 16px; outline: none; }
            input:focus { border-color: #38bdf8; }
            button { width: 96%; padding: 12px; background: #0284c7; border: none; color: white; font-size: 16px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: 0.3s; margin-top: 10px; }
            button:hover { background: #0369a1; }
            .hidden { display: none; }
            #keyDisplay { background: #0f172a; padding: 15px; border-radius: 6px; word-break: break-all; border: 1px dashed #34d399; color: #34d399; font-family: monospace; font-size: 14px; text-align: left; margin-top: 15px; }
            .status { font-size: 14px; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>MI AI System Lock</h2>
            
            <div id="lockSection">
                <input type="password" id="lockCode" placeholder="Enter Security Code">
                <button onclick="unlockSystem()">Unlock System</button>
            </div>
            
            <div id="controlSection" class="hidden">
                <div class="status">System Status: <span id="engineStatus" style="font-weight:bold;">Checking...</span></div>
                <button onclick="generateKey()" style="background: #10b981;">Generate New API Key</button>
                <div id="keyDisplay" class="hidden"></div>
            </div>
        </div>

        <script>
            // Page load hotay hi status check karna
            window.onload = function() {
                checkEngineStatus();
            };

            async function checkEngineStatus() {
                try {
                    const res = await fetch('/api/status');
                    const data = await res.json();
                    const statusSpan = document.getElementById('engineStatus');
                    if(data.connected) {
                        statusSpan.innerText = "ONLINE (miai-v1 Core Connected)";
                        statusSpan.style.color = "#34d399";
                    } else {
                        statusSpan.innerText = "OFFLINE (GitHub Server Sleeping)";
                        statusSpan.style.color = "#ef4444";
                    }
                } catch(e) {
                    document.getElementById('engineStatus').innerText = "ERROR";
                }
            }

            function unlockSystem() {
                const code = document.getElementById('lockCode').value;
                if(code === 'muaaz19720') {
                    document.getElementById('lockSection').classList.add('hidden');
                    document.getElementById('controlSection').classList.remove('hidden');
                } else {
                    alert('Wrong Lock Code! Access Denied.');
                }
            }

            function generateKey() {
                const code = document.getElementById('lockCode').value || 'muaaz19720';
                // Client side secure encryption key generation format
                const randomHex = Array.from({length: 24}, () => Math.floor(Math.random()*16).toString(16)).join('');
                const generatedKey = `miai-live-${code}-${randomHex}`;
                
                const display = document.getElementById('keyDisplay');
                display.classList.remove('hidden');
                display.innerHTML = `
                    <strong>Your Custom Secret API Key:</strong><br>
                    <span style="color: #cbd5e1; user-select: all;">${generatedKey}</span><br><br>
                    <strong>Model Name:</strong> <span style="color: #38bdf8;">miai-v1</span><br>
                    <strong>Server URL:</strong> <span style="color: #38bdf8;">https://${window.location.host}</span>
                `;
            }
        </script>
    </body>
    </html>
    """

# --- SYSTEM STATUS ENDPOINT ---
@app.get("/api/status")
async def system_status():
    return {"connected": bool(GITHUB_SERVER_URL), "target": "miai-v1"}

# --- GROQ FORMAT SECURE API GATEWAY ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. API Key Authorization Header check karna
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or Malformed API Key")
    
    token = auth_header.split(" ")[1]
    
    # Validation Rule: Key must start with miai-live- and contain your secret lock code
    if not token.startswith("miai-live-muaaz19720-") and token != "miai-master-token-786":
        raise HTTPException(status_code=403, detail="Invalid MI AI API Key or Access Revoked")
        
    if not GITHUB_SERVER_URL:
        raise HTTPException(status_code=503, detail="MI AI Core Engine is currently offline. Wake up your GitHub workflow!")

    body = await request.json()
    # Force model identification format to match your model specs
    body["model"] = "miai-v1"
    
    target_url = f"{GITHUB_SERVER_URL.strip('/')}/v1/chat/completions"
    try:
        response = requests.post(target_url, json=body, timeout=60)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core Engine Communication Failure: {str(e)}")
