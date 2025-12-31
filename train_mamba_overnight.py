import os
import json
import torch
import random
import math
import re
import argparse
from transformers import AutoTokenizer, MambaForCausalLM
from datasets import load_dataset
import time
import logging
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--resume", nargs="?", const=True, default=False, 
                    help="Resume from checkpoint. Optionally specify checkpoint dir.")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HUGGINGFACE", os.environ.get("HF_TOKEN"))
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")

FIM_PREFIX = "«PREFIX»"
FIM_SUFFIX = "«SUFFIX»"
FIM_MIDDLE = "«MIDDLE»"
CURSOR_CTX = "«CTX»"
CURSOR = "«CURSOR»"
AFTER_CURSOR = "«/CTX»"

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

resume_checkpoint = None
resume_step = 0
if args.resume:
    if args.resume is True:
        resume_checkpoint = get_latest_checkpoint()
    else:
        resume_checkpoint = Path(args.resume)
        if not resume_checkpoint.exists():
            logger.error(f"Checkpoint not found: {resume_checkpoint}")
            exit(1)
    
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        meta_file = resume_checkpoint / "training_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                resume_step = meta.get("step", 0)
                logger.info(f"Resuming from step {resume_step}")

if resume_checkpoint:
    logger.info("Loading model and tokenizer from checkpoint...")
    model = MambaForCausalLM.from_pretrained(
        resume_checkpoint, 
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(resume_checkpoint, token=HF_TOKEN)
else:
    logger.info("Loading model and tokenizer...")
    model = MambaForCausalLM.from_pretrained(
        "state-spaces/mamba-130m-hf", 
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, CURSOR_CTX, CURSOR, AFTER_CURSOR]})
    model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info(f"Parameters: {num_params:.1f}M")

URL_PATTERN = re.compile(r'https?://[^\s\'">\]})]+|www\.[^\s\'">\]})]+')

BANNED_LANGUAGES = {
    'csv', 'tsv', 'json', 'yaml', 'toml', 'xml', 'svg',
    'markdown', 'restructuredtext', 'text', 'diff', 'sql',
    'jupyter notebook', 'jupyter-notebook', 'turtle', 'api blueprint',
}

CONTEXT_LINES = 10
CORRUPTION_PROB = 0.0
INCOMPLETE_PROB = 0.35

logger.info("Loading The Stack (Python only) streaming dataset...")
dataset = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",
    split="train",
    streaming=True,
    token=HF_TOKEN
)

batch_size = 12
max_length = 256
gradient_accumulation_steps = 4
effective_batch = batch_size * gradient_accumulation_steps
learning_rate = 1e-4
warmup_steps = 100
log_interval = 50
save_interval = 500
max_checkpoints = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scaler = torch.amp.GradScaler()

loss_history = []
best_loss = float('inf')
checkpoint_losses = []

lang_stats = {"processed": {}, "skipped_blacklist": {}}
LANG_STATS_FILE = Path("lang_distribution.log")

def update_lang_stats(lang, processed=True):
    category = "processed" if processed else "skipped_blacklist"
    lang_stats[category][lang] = lang_stats[category].get(lang, 0) + 1

def write_lang_stats():
    with open(LANG_STATS_FILE, "w") as f:
        f.write(f"Language Distribution Stats (updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PROCESSED FILES BY LANGUAGE:\n")
        f.write("-" * 40 + "\n")
        total_processed = sum(lang_stats["processed"].values())
        for lang, count in sorted(lang_stats["processed"].items(), key=lambda x: -x[1]):
            pct = (count / total_processed * 100) if total_processed > 0 else 0
            f.write(f"  {lang:30} {count:8,} ({pct:5.2f}%)\n")
        f.write(f"  {'TOTAL':30} {total_processed:8,}\n\n")
        
        f.write("SKIPPED FILES (BLACKLISTED LANGUAGES):\n")
        f.write("-" * 40 + "\n")
        total_skipped = sum(lang_stats["skipped_blacklist"].values())
        for lang, count in sorted(lang_stats["skipped_blacklist"].items(), key=lambda x: -x[1]):
            pct = (count / total_skipped * 100) if total_skipped > 0 else 0
            f.write(f"  {lang:30} {count:8,} ({pct:5.2f}%)\n")
        f.write(f"  {'TOTAL':30} {total_skipped:8,}\n\n")
        
        grand_total = total_processed + total_skipped
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total processed:  {total_processed:,}\n")
        f.write(f"  Total skipped:    {total_skipped:,}\n")
        f.write(f"  Grand total:      {grand_total:,}\n")
        if grand_total > 0:
            f.write(f"  Skip rate:        {total_skipped/grand_total*100:.2f}%\n")

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    return learning_rate

def get_safe_split_range(line):
    match = URL_PATTERN.search(line)
    if match:
        return (0, match.start())
    return (0, len(line))

def weighted_split_point(line):
    min_pos, max_pos = get_safe_split_range(line)
    if max_pos <= min_pos:
        return 0
    length = max_pos - min_pos
    weights = [1.0 - (i / (length + 1)) * 0.5 for i in range(length + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return min_pos + random.choices(range(length + 1), weights=probs)[0]

def apply_fim(content):
    lines = content.split('\n')
    if len(lines) < 2:
        return None
    
    line_idx = random.randint(0, len(lines) - 1)
    line = lines[line_idx]
    
    split_pos = weighted_split_point(line)
    
    prefix_lines = lines[:line_idx]
    prefix_partial = line[:split_pos]
    prefix = '\n'.join(prefix_lines)
    if prefix_lines:
        prefix += '\n'
    prefix += prefix_partial
    
    if split_pos >= len(line):
        if line_idx + 1 >= len(lines):
            return None
        middle = lines[line_idx + 1] + '\n'
        suffix_lines = lines[line_idx + 2:]
        suffix_start_line_idx = line_idx + 2
    else:
        middle = line[split_pos:] + '\n'
        suffix_lines = lines[line_idx + 1:]
        suffix_start_line_idx = line_idx + 1
    
    if not middle.strip():
        return None
    
    should_truncate = random.random() < INCOMPLETE_PROB
    
    suffix = '\n'.join(suffix_lines)
    
    if should_truncate:
        suffix = ''
    
    ctx_before_start = max(0, line_idx - CONTEXT_LINES)
    ctx_before_lines = lines[ctx_before_start:line_idx]
    ctx_before = '\n'.join(ctx_before_lines)
    if ctx_before_lines:
        ctx_before += '\n'
    ctx_before += prefix_partial
    
    ctx_after_end = min(len(lines), suffix_start_line_idx + CONTEXT_LINES)
    ctx_after_lines = lines[suffix_start_line_idx:ctx_after_end]
    
    if should_truncate:
        ctx_after = ''
    else:
        ctx_after = '\n'.join(ctx_after_lines)
    
    fim_string = f"{CURSOR_CTX}{ctx_before}{CURSOR}{ctx_after}{AFTER_CURSOR}{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    
    return fim_string

fim_ratio = 1.0

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

def tokenize_texts(texts):
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return tokens["input_ids"]

def save_checkpoint(step, avg_loss, reason="scheduled"):
    global checkpoint_losses
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"ckpt_step{step}_{timestamp}"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    
    metadata = {
        "step": step,
        "loss": avg_loss,
        "timestamp": timestamp,
        "reason": reason,
        "config": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch": effective_batch,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "model": "state-spaces/mamba-370m-hf",
            "dataset": "bigcode/the-stack-dedup (all languages)",
        },
        "loss_history_last_100": loss_history[-100:] if loss_history else [],
    }
    
    with open(ckpt_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    checkpoint_losses.append((ckpt_path, avg_loss, step))
    
    if len(checkpoint_losses) > max_checkpoints:
        checkpoint_losses.sort(key=lambda x: x[1])
        while len(checkpoint_losses) > max_checkpoints:
            worst = checkpoint_losses.pop()
            if worst[0].exists():
                import shutil
                shutil.rmtree(worst[0])
                logger.info(f"Removed old checkpoint: {worst[0].name} (loss={worst[1]:.4f})")
    
    logger.info(f"Saved checkpoint to {ckpt_name} (loss={avg_loss:.4f}, reason={reason})")

logger.info(f"Config: batch={batch_size}, grad_accum={gradient_accumulation_steps}, eff_batch={effective_batch}, lr={learning_rate}")
logger.info(f"Checkpoints: every {save_interval} steps, keeping best {max_checkpoints}")
logger.info("Starting training (runs forever until interrupted)...")

model.train()
total_loss = 0
step = resume_step
accum_step = 0
start_time = time.time()
epoch = 0
samples_seen = 0
running_loss = None
fim_buffer = []

dataset_iter = iter(dataset)

try:
    while True:
        while len(fim_buffer) < batch_size:
            try:
                example = next(dataset_iter)
            except StopIteration:
                epoch += 1
                logger.info(f"Completed epoch {epoch}, restarting dataset...")
                dataset_iter = iter(dataset)
                example = next(dataset_iter)
            
            content = example.get("content", "")
            lang = example.get("lang", "unknown").lower()
            
            if lang in BANNED_LANGUAGES:
                update_lang_stats(lang, processed=False)
                continue
            
            if len(content.strip()) < 50:
                continue
            
            if content.count('\n') < 3:
                continue
            
            fim_samples = generate_fim_samples(content)
            fim_buffer.extend(fim_samples)
            samples_seen += 1
            update_lang_stats(lang, processed=True)
        
        batch_texts = fim_buffer[:batch_size]
        fim_buffer = fim_buffer[batch_size:]
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)
        
        input_ids = tokenize_texts(batch_texts).to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / gradient_accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected at step {step}! Skipping batch.")
            optimizer.zero_grad()
            accum_step = 0
            continue
        
        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        accum_step += 1
        
        if accum_step >= gradient_accumulation_steps:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"NaN/Inf gradient at step {step}! Skipping update.")
                optimizer.zero_grad()
                scaler.update()
                accum_step = 0
                total_loss = 0
                continue
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accum_step = 0
            step += 1
            
            step_loss = total_loss / gradient_accumulation_steps
            loss_history.append(step_loss)
            if len(loss_history) > 1000:
                loss_history.pop(0)
            
            if running_loss is None:
                running_loss = step_loss
            else:
                running_loss = 0.95 * running_loss + 0.05 * step_loss
            
            total_loss = 0
            
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = log_interval / elapsed
                samples_per_sec = steps_per_sec * effective_batch
                mem = torch.cuda.max_memory_allocated() / 1e9
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Step {step} | epoch {epoch} | loss={step_loss:.4f} (ema={running_loss:.4f}) | "
                    f"lr={current_lr:.2e} | {samples_per_sec:.1f} samp/s | mem={mem:.2f}GB | "
                    f"samples={samples_seen:,}"
                )
                
                if running_loss < best_loss:
                    best_loss = running_loss
                
                start_time = time.time()
            
            if step % save_interval == 0:
                save_checkpoint(step, running_loss, reason="scheduled")
                write_lang_stats()

except KeyboardInterrupt:
    logger.info("Training interrupted by user (Ctrl+C)")
    save_checkpoint(step, running_loss or 0, reason="interrupted")
    write_lang_stats()
except Exception as e:
    logger.error(f"Training failed with error: {e}")
    save_checkpoint(step, running_loss or 0, reason=f"error: {str(e)[:50]}")
    write_lang_stats()
    raise

logger.info(f"Training ended. Total steps: {step}, samples: {samples_seen:,}, best_loss: {best_loss:.4f}")
