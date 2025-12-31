import os
import random
import math
from datasets import load_dataset

HF_TOKEN = os.environ.get("HUGGINGFACE", os.environ.get("HF_TOKEN"))

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

def weighted_split_point(line_length):
    if line_length == 0:
        return 0
    weights = [1.0 - (i / (line_length + 1)) * 0.5 for i in range(line_length + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(line_length + 1), weights=probs)[0]

def apply_fim(content):
    lines = content.split('\n')
    if len(lines) < 2:
        return None
    
    line_idx = random.randint(0, len(lines) - 1)
    line = lines[line_idx]
    
    split_pos = weighted_split_point(len(line))
    
    prefix_lines = lines[:line_idx]
    prefix_partial = line[:split_pos]
    prefix = '\n'.join(prefix_lines)
    if prefix_lines:
        prefix += '\n'
    prefix += prefix_partial
    
    if split_pos >= len(line):
        if line_idx + 1 >= len(lines):
            return None
        middle = lines[line_idx + 1]
        suffix_lines = lines[line_idx + 2:]
    else:
        middle = line[split_pos:]
        suffix_lines = lines[line_idx + 1:]
    
    suffix = '\n'.join(suffix_lines)
    
    if not middle.strip():
        return None
    
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"

def generate_fim_samples(content, max_samples=200):
    lines = content.split('\n')
    if len(lines) < 2:
        return []
    
    num_samples = min(len(lines) // 5, max_samples)
    num_samples = max(1, num_samples)
    
    samples = []
    attempts = 0
    while len(samples) < num_samples and attempts < num_samples * 3:
        result = apply_fim(content)
        attempts += 1
        if result:
            samples.append(result)
    return samples

print("Loading dataset...")
dataset = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",
    split="train",
    streaming=True,
    token=HF_TOKEN
)

print("\n" + "="*80)
print("Testing FIM transformation on random samples")
print("="*80)

dataset_iter = iter(dataset)
for i in range(3):
    example = next(dataset_iter)
    content = example["content"]
    lines = content.split('\n')
    
    print(f"\n\n{'='*80}")
    print(f"SAMPLE {i+1}: {len(lines)} lines, {len(content)} chars")
    print("="*80)
    
    print("\n--- ORIGINAL (first 20 lines) ---")
    for j, line in enumerate(lines[:20]):
        print(f"{j+1:3}: {line[:100]}")
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    fim_result = apply_fim(content)
    if fim_result:
        print("\n--- FIM TRANSFORMED ---")
        parts = fim_result.split(FIM_SUFFIX)
        prefix_part = parts[0].replace(FIM_PREFIX, "")
        rest = parts[1].split(FIM_MIDDLE)
        suffix_part = rest[0]
        middle_part = rest[1]
        
        print(f"\nPREFIX (last 5 lines):")
        prefix_lines = prefix_part.split('\n')
        for line in prefix_lines[-5:]:
            print(f"  {line[:100]}")
        
        print(f"\nMIDDLE (what model should predict):")
        print(f"  >>> {middle_part[:100]} <<<")
        
        print(f"\nSUFFIX (first 3 lines):")
        suffix_lines = suffix_part.split('\n')
        for line in suffix_lines[:3]:
            print(f"  {line[:100]}")
        
        print(f"\n[Full FIM length: {len(fim_result)} chars]")

print("\n\n" + "="*80)
print("Testing split point distribution (100 samples on 50-char line)")
print("="*80)
print("Linear weighting: start of line = 1.0, end of line = 0.5")
positions = [weighted_split_point(50) for _ in range(100)]
for bucket in range(0, 51, 10):
    count = sum(1 for p in positions if bucket <= p < bucket + 10)
    print(f"  Positions {bucket:2}-{bucket+9:2}: {'#' * count} ({count}%)")

print("\n\n" + "="*80)
print("Testing multi-sample generation on longer file")
print("="*80)
for example in dataset_iter:
    content = example["content"]
    lines = content.split('\n')
    if len(lines) > 100:
        print(f"Found file with {len(lines)} lines")
        print(f"Expected samples: {min(len(lines) // 5, 200)}")
        samples = generate_fim_samples(content)
        print(f"Generated {len(samples)} FIM samples")
        for j, s in enumerate(samples[:3]):
            middle_start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
            middle = s[middle_start:middle_start+60]
            print(f"  Sample {j+1} middle: {middle}...")
        break
