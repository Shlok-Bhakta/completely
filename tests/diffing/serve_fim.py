from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from pathlib import Path
import json
import uvicorn
import argparse

from data_analysis import format_inference_prompt, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true", help="Use CUDA GPU")
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

app = FastAPI()

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

model = None
tokenizer = None

def get_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    checkpoints = list(CHECKPOINT_DIR.iterdir())
    if not checkpoints:
        return None
    latest = None
    latest_step = -1
    for ckpt in checkpoints:
        meta_file = ckpt / "training_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                step = meta.get("step", -1)
                if step > latest_step:
                    latest_step = step
                    latest = ckpt
    return latest

@app.on_event("startup")
def load_model():
    global model, tokenizer
    ckpt = get_latest_checkpoint()
    if ckpt is None:
        print("No checkpoint found!")
        return
    print(f"Loading checkpoint: {ckpt}")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = MambaForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16 if args.cuda else torch.float32)
    model = model.to(device)
    model.eval()
    print("Model loaded!")

def run_inference(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    middle_start = generated.find(FIM_MIDDLE)
    if middle_start != -1:
        completion = generated[middle_start + len(FIM_MIDDLE):]
        for end_token in [FIM_PREFIX, FIM_SUFFIX, "<|endoftext|>", tokenizer.eos_token]:
            end_pos = completion.find(end_token)
            if end_pos > 0:
                completion = completion[:end_pos]
                break
        return completion
    return ""

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    suffix = body.get("suffix", "")
    max_tokens = body.get("max_tokens", 50)
    
    prefix_text = prompt[-1000:] if len(prompt) > 1000 else prompt
    suffix_text = suffix[:500] if len(suffix) > 500 else suffix
    
    fim_prompt = format_inference_prompt(prefix_text, suffix_text)
    
    completion = run_inference(fim_prompt, max_new_tokens=max_tokens)
    
    return JSONResponse({
        "id": "cmpl-local",
        "object": "text_completion",
        "choices": [{
            "text": completion,
            "index": 0,
            "finish_reason": "stop"
        }]
    })

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/v1/health")
def health_v1():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/v1/models")
def models():
    return {"data": [{"id": "local-fim", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 50)
    
    prompt = ""
    for msg in messages:
        prompt += msg.get("content", "")
    
    prefix_text = prompt[-1000:] if len(prompt) > 1000 else prompt
    fim_prompt = format_inference_prompt(prefix_text, "")
    
    completion = run_inference(fim_prompt, max_new_tokens=max_tokens)
    
    return JSONResponse({
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": completion},
            "index": 0,
            "finish_reason": "stop"
        }]
    })

@app.post("/api/generate")
async def ollama_generate(request: Request):
    body = await request.json()
    print(f"[/api/generate] Received: {json.dumps(body, indent=2)[:500]}")
    prompt = body.get("prompt", "")
    suffix = body.get("suffix", "")
    
    prefix_text = prompt[-1000:] if len(prompt) > 1000 else prompt
    suffix_text = suffix[:500] if len(suffix) > 500 else suffix
    fim_prompt = format_inference_prompt(prefix_text, suffix_text)
    
    completion = run_inference(fim_prompt, max_new_tokens=50)
    print(f"[/api/generate] Response: {completion[:100]}")
    
    return JSONResponse({
        "model": body.get("model", "local-fim"),
        "response": completion,
        "done": True
    })

@app.get("/api/tags")
def ollama_tags():
    return {"models": [{"name": "local-fim", "modified_at": "2024-01-01T00:00:00Z"}]}

if __name__ == "__main__":
    print("Server running on http://localhost:8000")
    print("Endpoints:")
    print("  POST /v1/completions     - OpenAI completions")
    print("  POST /v1/chat/completions - OpenAI chat")
    print("  POST /api/generate       - Ollama format")
    uvicorn.run(app, host="0.0.0.0", port=8000)
