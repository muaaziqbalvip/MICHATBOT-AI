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

app = FastAPI(title="MI AI Core Engine")
MODEL_PATH = "./model_files"

auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token or auth_token.strip() == "":
    print("\nCRITICAL ERROR: NGROK_AUTH_TOKEN is empty!")
    sys.exit(1)

print("Booting miai-v1 Engine...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
print("miai-v1 Model Loaded Successfully into GitHub RAM!")

# Ekdum saaf aur clear system prompt
SYSTEM_PROMPT = (
    "Aapka naam 'miai-v1' hai. Aap ek pro-level advanced AI assistant hain. "
    "Aapko aapke creator Muaaz ne design aur develop kiya hai. Aap hamesha Muaaz ke rules follow karte hain. "
    "Hamesha user ke sawal ka aqalmandi se Roman Urdu ya Hindi mein jawab dein."
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "miai-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 250

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    try:
        # ChatML format structure strictly for SmolLM2
        full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        
        for msg in request.messages:
            full_prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
            
        full_prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens, 
                temperature=request.temperature, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Sirf naya generate hua text nikalna (Input prompt ko minus karna)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Agar model phir bhi tag include kare toh use safae se urana
        if "<|im_end|>" in response_content:
            response_content = response_content.split("<|im_end|>")[0].strip()

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
    try:
        ngrok.set_auth_token(auth_token)
        public_url = ngrok.connect(8000)
        print(f"\n[ENGINE LIVE] Public Ngrok Tunnel: {public_url.public_url}\n")
    except exception.PyngrokNgrokError as e:
        print(f"Ngrok Boot Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
