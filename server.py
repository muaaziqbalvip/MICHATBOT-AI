import os
import sys
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok, exception
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="MI AI Qwen Fixed Engine")
MODEL_PATH = "./model_files"

auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token or auth_token.strip() == "":
    print("\nCRITICAL ERROR: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

print("Booting Fixed Qwen Engine...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
print("Qwen Model Loaded 100%!")

# Ekdum simple system prompt
SYSTEM_PROMPT = "Aap ek AI assistant hain. Hamesha user ke sawal ka short aur clear jawab Roman Urdu ya Hindi me dein."

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "miai-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.3  # Isko kam rakha hai taaki pagal na ho
    max_tokens: Optional[int] = 100

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    try:
        # --- QWEN OFFICIAL HARDCODED CHATML FORMAT ---
        # Qwen model is structure ko 100% samajhta hai
        full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        
        for msg in request.messages:
            # Agar system prompt user dobara bhej raha ho toh bypass karein
            if msg.role == "system":
                continue
            full_prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
            
        full_prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens, 
                temperature=request.temperature, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Sirf naya answer cut karna
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Safai: Agar model extra tags print kare toh urana
        if "<|im_end|>" in response_content:
            response_content = response_content.split("<|im_end|>")[0].strip()
        if "assistant" in response_content:
            response_content = response_content.replace("assistant", "").strip()

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
        print(f"\n[ENGINE LIVE] URL: {public_url.public_url}\n")
    except exception.PyngrokNgrokError as e:
        print(f"Ngrok Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok)
    t.daemon = True
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
