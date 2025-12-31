import os
import time
from pathlib import Path
from transformers import AutoTokenizer, MambaForCausalLM
import torch

CHECKPOINT_DIR = Path("checkpoints")

FIM_PREFIX = "«PREFIX»"
FIM_SUFFIX = "«SUFFIX»"
FIM_MIDDLE = "«MIDDLE»"
CURSOR_CTX = "«CTX»"
CURSOR = "«CURSOR»"
AFTER_CURSOR = "«/CTX»"

TEST_SAMPLES = [
    {
        "prefix": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
        "suffix": "\n\nprint(fibonacci(10))",
        "description": "Fibonacci recursive call"
    },
    {
        "prefix": "import os\n\ndef list_files(directory):\n    ",
        "suffix": "\n\nlist_files('.')",
        "description": "List files function body"
    },
    {
        "prefix": "class User:\n    def __init__(self, name, email):\n        self.name = ",
        "suffix": "\n        self.email = email",
        "description": "Class attribute assignment"
    },
    {
        "prefix": "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        ",
        "suffix": "\n            return await response.json()",
        "description": "Async HTTP request"
    },
]

def get_latest_checkpoint():
    """Get the most recent checkpoint by step number."""
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
            import json
            with open(meta_file) as f:
                meta = json.load(f)
                step = meta.get("step", -1)
                if step > latest_step:
                    latest_step = step
                    latest = (ckpt, meta)
    
    return latest

def run_inference(model, tokenizer, prefix, suffix, max_new_tokens=50):
    # Simple cursor context: last 2 lines of prefix
    prefix_lines = prefix.split('\n')
    ctx_before = '\n'.join(prefix_lines[-2:]) if len(prefix_lines) >= 2 else prefix
    
    # Simple cursor context: first 2 lines of suffix
    suffix_lines = suffix.split('\n')
    ctx_after = '\n'.join(suffix_lines[:2]) if len(suffix_lines) >= 2 else suffix
    
    prompt = f"{CURSOR_CTX}{ctx_before}{CURSOR}{ctx_after}{AFTER_CURSOR}{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"  Raw output: {generated[-200:]}")
    
    for middle_token in [FIM_MIDDLE, "MIDDLE"]:
        middle_start = generated.find(middle_token)
        if middle_start != -1:
            completion = generated[middle_start + len(middle_token):]
            for end_token in [FIM_PREFIX, "PREFIX", "<|endoftext|>", "\n\n"]:
                end_pos = completion.find(end_token)
                if end_pos > 0:
                    completion = completion[:end_pos]
                    break
            return completion.strip() if completion.strip() else "(empty)"
    return "(no fim_middle found)"

def main():
    print("=" * 60)
    print("FIM Model Monitor - Checking every 5 minutes")
    print("=" * 60)
    
    current_ckpt = None
    model = None
    tokenizer = None
    
    while True:
        best = get_latest_checkpoint()
        
        if best is None:
            print(f"\n[{time.strftime('%H:%M:%S')}] No checkpoints found yet...")
            time.sleep(300)
            continue
        
        ckpt_path, meta = best
        
        if current_ckpt != ckpt_path:
            print(f"\n[{time.strftime('%H:%M:%S')}] Loading new checkpoint: {ckpt_path.name}")
            print(f"  Step: {meta.get('step', '?')}, Loss: {meta.get('loss', '?'):.4f}")
            
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            model = MambaForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32)
            model.eval()
            current_ckpt = ckpt_path
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Running inference tests...")
        print("-" * 60)
        
        for sample in TEST_SAMPLES:
            print(f"\n{sample['description']}:")
            print(f"  Prefix: ...{sample['prefix'][-40:]}")
            completion = run_inference(model, tokenizer, sample['prefix'], sample['suffix'])
            print(f"  Completion: {completion}")
        
        print("\n" + "=" * 60)
        print("Sleeping 5 minutes...")
        time.sleep(300)

if __name__ == "__main__":
    main()
