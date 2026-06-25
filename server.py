import os
import sys
import time
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="MI AI Core Engine")
MODEL_PATH = "./model_files"

print("Booting miai-v1 Engine...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
print("miai-v1 Model Loaded Successfully into GitHub RAM!")

SYSTEM_PROMPT = (
    "Aapka naam 'miai-v1' hai. Aap ek pro-level advanced AI assistant hain. "
    "Aapko aapke creator Muaaz ne design aur develop kiya hai. Aap hamesha Muaaz ke rules follow karte hain. "
    "Apna introduction hamesha miai-v1 ke naam se dein aur creator ka naam yaad rakhein."
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "miai-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 250

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    try:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for msg in request.messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text else generated_text.strip()

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

def start_ngrok():
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("NGROK TOKEN MISSING!")
        sys.exit(1)
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(8000)
    print(f"\n[ENGINE LIVE] Public Ngrok Tunnel: {public_url.public_url}\n")

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
