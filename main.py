import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Phi-3.5 Mini API")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/phi-3.5-mini-instruct-q4_k_m.gguf")

# لود کردن مدل کوانتیزه
# n_ctx: حداکثر طول متن ورودی و خروجی (Context)
# n_threads: تعداد هسته‌های CPU که درگیر بشن
print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count() or 4,
    verbose=False
)
print("Model loaded successfully!")

class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: PromptRequest):
    # فرمت استاندارد پرامپت چت برای مدل Phi-3.5
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful and smart AI assistant."},
            {"role": "user", "content": req.prompt}
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    reply = response["choices"][0]["message"]["content"]
    return {"reply": reply}
