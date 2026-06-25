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

app = FastAPI(title="MI AI Pure Core")
MODEL_PATH = "./model_files"

auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token or auth_token.strip() == "":
    print("\nCRITICAL ERROR: NGROK_AUTH_TOKEN missing!")
    sys.exit(1)

print("Booting miai-v1 Pure Engine...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
print("miai-v1 Model Loaded!")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "miai-v1"
    messages: List[Message]
    temperature: Optional[float] = 0.3  # Temperature kam rakhne se model be-tuki baatein nahi karega
    max_tokens: Optional[int] = 80      # Chote aur fast answers ke liye tokens limit kam kar di

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    try:
        # Koi system prompt nahi, direct user ke messages ko template me dalna
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        # Official HuggingFace format apply karna bina kisi external instructions ke
        full_prompt = tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

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
            
        # Sirf aur sirf fresh generated raw text pick karna
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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
