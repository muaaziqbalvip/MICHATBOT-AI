import os
import sys
import time
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =====================================================================
# 1. INITIALIZATION (FastAPI App aur Model Path Configuration)
# =====================================================================
app = FastAPI(title="MI AI Core Engine")
MODEL_PATH = "./model_files"  # GitHub RAM ke andar cache folder ka path

# =====================================================================
# 2. SECURITY CHECK (Ngrok Token Verification)
# =====================================================================
# Yeh block check karta hai ki GitHub Secrets se token mila ya nahi.
# Agar token khali hoga toh script pehle hi band ho jayegi taaki crash na ho.
auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token or auth_token.strip() == "":
    print("\nCRITICAL ERROR: NGROK_AUTH_TOKEN is completely empty in GitHub Secrets!")
    sys.exit(1)

# =====================================================================
# 3. AI MODEL LOADING (Model aur Tokenizer RAM me Load Karna)
# =====================================================================
print("Booting miai-v1 Engine...")
# Tokenizer text ko numbers (tokens) me badalta hai taaki AI samajh sake
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Model main weights load karta hai cpu/torch float32 format me
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
print("miai-v1 Model Loaded Successfully into GitHub RAM!")

# =====================================================================
# 4. SYSTEM PROMPT CONFIGURATION (AI ki Personality/Identity)
# =====================================================================
# Yeh instructions model ke dimaag me fix rehti hain. AI ko isi ke mutabiq
# behave karna hota hai aur use pata hota hai ki uska boss Muaaz hai.
SYSTEM_PROMPT = (
    "Aapka naam 'miai-v1' hai. Aap ek pro-level advanced AI assistant hain. "
    "Aapko aapke creator Muaaz ne design aur develop kiya hai. Aap hamesha Muaaz ke rules follow karte hain. "
    "Hamesha user ke sawal ka aqalmandi se Roman Urdu ya Hindi mein short aur sweet jawab dein. "
    "Baat ko faltu lamba mat karein aur hamesha tameez se baat karein."
)

# =====================================================================
# 5. DATA SCHEMAS (Groq/OpenAI Standard Payload Format)
# =====================================================================
class Message(BaseModel):
    role: str      # 'user' ya 'assistant' ya 'system'
    content: str   # Asal text message

class ChatCompletionRequest(BaseModel):
    model: str = "miai-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.6  # Creative level (0.6 ekdum perfect aur stable hai)
    max_tokens: Optional[int] = 250     # Jawab ki maximum length

# =====================================================================
# 6. MAIN CHAT COMPLETION ENDPOINT (Yahan Asal Jadu Hota Hai)
# =====================================================================
@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    try:
        # ChatML Format Structure (SmolLM2 isi format ko samajhta hai)
        # Isse model ko pata chalta hai ki System Prompt kahan khatam hua aur User ka sawal kahan se shuru hua.
        full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        
        for msg in request.messages:
            full_prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
            
        # Assistant ka tag lagakar chhod dete hain taaki AI iske aage se jawab likhna shuru kare
        full_prompt += "<|im_start|>assistant\n"

        # Text prompt ko torch tensors (numbers format) me convert karna
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Jawab generate karne ka logic
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens, 
                temperature=request.temperature, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # --- FIXED REPETITION BUG ---
        # `inputs.input_ids.shape[1]` se hume pata chalta hai ki humara bheja gaya sawal kitna lamba tha.
        # `outputs[0][input_len:]` lagane se hum pichla poora sawal kaat dete hain aur sirf AI ka naya jawab uthate hain!
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        
        # Numbers ko wapas insani language (text) me decode karna
        response_content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Fail-safe cleaning: Agar model galti se chat tags end me print kare toh use saaf karna
        if "<|im_end|>" in response_content:
            response_content = response_content.split("<|im_end|>")[0].strip()

        # Groq / OpenAI standard output JSON format return karna
        return {
            "id": "chatcmpl-miai-v1",
            "object": "chat.completion",
            "model": "miai-v1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 7. BACKGROUND NGROK TUNNEL (Local Port ko Public URL Banana)
# =====================================================================
def start_ngrok():
    try:
        # Ngrok ko authenticate karna aapke token se
        ngrok.set_auth_token(auth_token)
        # Port 8000 (FastAPI port) ko internet par open karna
        public_url = ngrok.connect(8000)
        print(f"\n[ENGINE LIVE] Public Ngrok Tunnel Created: {public_url.public_url}\n")
    except exception.PyngrokNgrokError as e:
        print(f"Ngrok Boot Error: {str(e)}")
        sys.exit(1)

# =====================================================================
# 8. ENGINE RUNNER (Main Execution Start Block)
# =====================================================================
if __name__ == "__main__":
    # Ngrok ko ek alag parallel thread me chalana taaki API block na ho
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    
    # Uvicorn server ko local port 8000 par start karna
    uvicorn.run(app, host="0.0.0.0", port=8000)
