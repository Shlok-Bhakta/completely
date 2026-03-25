import json
import random
from pathlib import Path
from transformers import AutoTokenizer, MambaForCausalLM
import torch

from data_analysis import PythonFIMStackDataset, get_other_context

CHECKPOINT_DIR = Path("checkpoints")

FIM_PREFIX = "<prefix>"
FIM_SUFFIX = "<suffix>"
FIM_MIDDLE = "<middle>"
CURSOR = "<cursor>"
OTHER_CTX = "<context>"

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
                    latest = (ckpt, meta)
    
    return latest

def generate_test_samples(dataset_iter, n=100, incomplete=False):
    samples = []
    count = 0
    
    for _ in range(n):
        if count >= n:
            break
        try:
            fim_data, = next(dataset_iter)
            if incomplete:
                prompt = f"{CURSOR}{fim_data['cursor']}{OTHER_CTX}{fim_data['context']}{FIM_PREFIX}{fim_data['prefix']}{FIM_SUFFIX}{FIM_MIDDLE}"
            else:
                prompt = f"{CURSOR}{fim_data['cursor']}{OTHER_CTX}{fim_data['context']}{FIM_PREFIX}{fim_data['prefix']}{FIM_SUFFIX}{fim_data['suffix']}{FIM_MIDDLE}"
            samples.append({
                "prompt": prompt,
                "expected": fim_data['middle'],
                "filepath": "streaming",
                "incomplete": incomplete
            })
            count += 1
        except Exception as e:
            continue
    
    return samples

def run_inference(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs["input_ids"].ne(tokenizer.pad_token_id).long()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
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

def evaluate_samples(model, tokenizer, samples):
    correct = 0
    wrong_sample = None
    
    for sample in samples:
        completion = run_inference(model, tokenizer, sample["prompt"])
        expected = sample["expected"]
        
        if completion == expected:
            correct += 1
        elif wrong_sample is None:
            wrong_sample = {
                "filepath": sample["filepath"],
                "expected": expected,
                "got": completion,
                "incomplete": sample["incomplete"]
            }
    
    return correct, len(samples), wrong_sample

def main():
    print("=" * 60)
    print("FIM Model Accuracy Test")
    print("=" * 60)
    
    best = get_latest_checkpoint()
    
    if best is None:
        print("No checkpoints found!")
        return
    
    ckpt_path, meta = best
    print(f"Loading checkpoint: {ckpt_path.name}")
    print(f"Step: {meta.get('step', '?')}, Loss: {meta.get('loss', '?'):.4f}")
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = MambaForCausalLM.from_pretrained(ckpt_path, dtype=torch.float32)
    model.eval()
    
    print("\nLoading dataset from The Stack...")
    dataset = PythonFIMStackDataset()
    dataset_iter = iter(dataset)
    
    skip_count = random.randint(50, 500)
    print(f"Skipping {skip_count} samples for variety...")
    for _ in range(skip_count):
        try:
            next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
    
    print("\n--- Basic FIM (with suffix) ---")
    basic_samples = generate_test_samples(dataset_iter, n=1, incomplete=False)
    basic_correct, basic_total, basic_wrong = evaluate_samples(model, tokenizer, basic_samples)
    print(f"Accuracy: {basic_correct}/{basic_total} ({100*basic_correct/basic_total:.1f}%)")
    
    if basic_wrong:
        print(f"\nExample wrong completion (basic):")
        print(f"  File: {basic_wrong['filepath']}")
        print(f"  Expected: {repr(basic_wrong['expected'][:100])}")
        print(f"  Got:      {repr(basic_wrong['got'][:100])}")
    
    print("\n--- Incomplete FIM (no suffix) ---")
    incomplete_samples = generate_test_samples(dataset_iter, n=1, incomplete=True)
    inc_correct, inc_total, inc_wrong = evaluate_samples(model, tokenizer, incomplete_samples)
    print(f"Accuracy: {inc_correct}/{inc_total} ({100*inc_correct/inc_total:.1f}%)")
    
    if inc_wrong:
        print(f"\nExample wrong completion (incomplete):")
        print(f"  File: {inc_wrong['filepath']}")
        print(f"  Expected: {repr(inc_wrong['expected'][:100])}")
        print(f"  Got:      {repr(inc_wrong['got'][:100])}")
    
    print("\n" + "=" * 60)
    total_correct = basic_correct + inc_correct
    total_samples = basic_total + inc_total
    print(f"Overall: {total_correct}/{total_samples} ({100*total_correct/total_samples:.1f}%)")

if __name__ == "__main__":
    main()
