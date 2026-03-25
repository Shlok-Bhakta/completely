import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import random
import argparse
from transformers import AutoTokenizer, MambaForCausalLM
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

from data_analysis import PythonFIMStackDataset, format_fim_string, get_special_tokens

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
    tokenizer.add_special_tokens({"additional_special_tokens": get_special_tokens()})
    model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()
model = model.to(device)
logger.info("Compiling model (this may take a few minutes)...")
# model = torch.compile(model)
logger.info("Model compiled!")

num_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info(f"Parameters: {num_params:.1f}M")

INCOMPLETE_PROB = 0.35

fim_dataset = PythonFIMStackDataset()
fim_iter = iter(fim_dataset)

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
scaler = torch.amp.GradScaler("cuda")

loss_history = []
best_loss = float('inf')
checkpoint_losses = []

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    return learning_rate

def get_fim_batch(size):
    samples = []
    for _ in range(size):
        try:
            item = next(fim_iter)
            samples.append(format_fim_string(item[0]))
        except StopIteration:
            break
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
            "model": "state-spaces/mamba-130m-hf",
            "dataset": "django codebase",
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
                shutil.rmtree(worst[0])
                logger.info(f"Removed old checkpoint: {worst[0].name} (loss={worst[1]:.4f})")
    
    logger.info(f"Saved checkpoint to {ckpt_name} (loss={avg_loss:.4f}, reason={reason})")

logger.info(f"Config: batch={batch_size}, grad_accum={gradient_accumulation_steps}, eff_batch={effective_batch}, lr={learning_rate}")
logger.info(f"Checkpoints: every {save_interval} steps, keeping best {max_checkpoints}")
logger.info(f"Training on bigcode/the-stack-dedup Python dataset (streaming)")
logger.info("Starting training (runs forever until interrupted)...")

model.train()
total_loss = 0
step = resume_step
accum_step = 0
start_time = time.time()
samples_seen = 0
running_loss = None

try:
    while True:
        batch_texts = get_fim_batch(batch_size)
        if not batch_texts:
            logger.warning("Dataset exhausted, this shouldn't happen with streaming")
            break
        
        samples_seen += len(batch_texts)
        
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
                    f"Step {step} | samples={samples_seen} | loss={step_loss:.4f} (ema={running_loss:.4f}) | "
                    f"lr={current_lr:.2e} | {samples_per_sec:.1f} samp/s | mem={mem:.2f}GB"
                )
                
                if running_loss < best_loss:
                    best_loss = running_loss
                
                start_time = time.time()
            
            if step % save_interval == 0:
                save_checkpoint(step, running_loss, reason="scheduled")

except KeyboardInterrupt:
    logger.info("Training interrupted by user (Ctrl+C)")
    save_checkpoint(step, running_loss or 0, reason="interrupted")
except Exception as e:
    logger.error(f"Training failed with error: {e}")
    save_checkpoint(step, running_loss or 0, reason=f"error: {str(e)[:50]}")
    raise

logger.info(f"Training ended. Total steps: {step}, samples: {samples_seen}, best_loss: {best_loss:.4f}")
