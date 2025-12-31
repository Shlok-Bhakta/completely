#!/usr/bin/env python3
"""
FIM (Fill-in-Middle) Visualizer

Controls:
  n - Next sample
  p - Previous sample
  r - Regenerate (resample same file)
  q - Quit
"""

import os
import random
import re
import sys
import tty
import termios
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

URL_PATTERN = re.compile(r'https?://[^\s\'">\]})]+|www\.[^\s\'">\]})]+')

BANNED_LANGUAGES = {
    'csv', 'tsv', 'json', 'yaml', 'toml', 'xml', 'svg',
    'markdown', 'restructuredtext', 'text', 'diff', 'sql',
    'jupyter notebook', 'jupyter-notebook', 'turtle', 'api blueprint',
}

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    PURPLE = "\033[95m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def red(text: str) -> str:
    return f"{Colors.RED}{text}{Colors.RESET}"

def green(text: str) -> str:
    return f"{Colors.GREEN}{text}{Colors.RESET}"

def purple(text: str) -> str:
    return f"{Colors.PURPLE}{text}{Colors.RESET}"

def yellow(text: str) -> str:
    return f"{Colors.YELLOW}{text}{Colors.RESET}"

def cyan(text: str) -> str:
    return f"{Colors.CYAN}{text}{Colors.RESET}"

def gray(text: str) -> str:
    return f"{Colors.GRAY}{text}{Colors.RESET}"

def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"

FIM_PREFIX = "«PREFIX»"
FIM_SUFFIX = "«SUFFIX»"
FIM_MIDDLE = "«MIDDLE»"
CURSOR_CTX = "«CTX»"
CURSOR = "«CURSOR»"
AFTER_CURSOR = "«/CTX»"

CONTEXT_LINES = 10
CORRUPTION_PROB = 0.0
INCOMPLETE_PROB = 0.35

def corrupt_text(text: str) -> str:
    if not text or len(text) < 2:
        return text
    
    text_chars = list(text)
    corruption_type = random.choice(['swap', 'bracket', 'typo', 'duplicate', 'delete'])
    
    if corruption_type == 'swap' and len(text_chars) >= 2:
        idx = random.randint(0, len(text_chars) - 2)
        text_chars[idx], text_chars[idx + 1] = text_chars[idx + 1], text_chars[idx]
    
    elif corruption_type == 'bracket':
        brackets = {'(': '[', ')': ']', '[': '{', ']': '}', '{': '(', '}': ')'}
        bracket_indices = [i for i, c in enumerate(text_chars) if c in brackets]
        if bracket_indices:
            idx = random.choice(bracket_indices)
            text_chars[idx] = brackets[text_chars[idx]]
    
    elif corruption_type == 'typo':
        alpha_indices = [i for i, c in enumerate(text_chars) if c.isalpha()]
        if alpha_indices:
            idx = random.choice(alpha_indices)
            original = text_chars[idx]
            if original.islower():
                nearby = 'qwertyuiopasdfghjklzxcvbnm'
            else:
                nearby = 'QWERTYUIOPASDFGHJKLZXCVBNM'
            text_chars[idx] = random.choice(nearby)
    
    elif corruption_type == 'duplicate' and len(text_chars) > 0:
        idx = random.randint(0, len(text_chars) - 1)
        text_chars.insert(idx, text_chars[idx])
    
    elif corruption_type == 'delete' and len(text_chars) > 1:
        idx = random.randint(0, len(text_chars) - 1)
        text_chars.pop(idx)
    
    return ''.join(text_chars)

def get_safe_split_range(line: str) -> tuple[int, int]:
    match = URL_PATTERN.search(line)
    if match:
        return (0, match.start())
    return (0, len(line))

def weighted_split_point(line: str) -> int:
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
    
    rand_val = random.random()
    should_corrupt = rand_val < CORRUPTION_PROB
    should_truncate = not should_corrupt and rand_val < (CORRUPTION_PROB + INCOMPLETE_PROB)
    
    middle_corrupted = middle
    
    if should_corrupt:
        middle_without_newline = middle.rstrip('\n')
        if middle_without_newline:
            corrupted = corrupt_text(middle_without_newline)
            middle_corrupted = corrupted + '\n'
    
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
    
    if should_corrupt:
        ctx_after = middle_corrupted.rstrip('\n')
        if ctx_after_lines:
            ctx_after += '\n' + '\n'.join(ctx_after_lines)
    elif should_truncate:
        ctx_after = ''
    else:
        ctx_after = '\n'.join(ctx_after_lines)
    
    fim_suffix_final = suffix
    if should_corrupt:
        fim_suffix_final = middle_corrupted
    
    fim_string = f"{CURSOR_CTX}{ctx_before}{CURSOR}{ctx_after}{AFTER_CURSOR}{FIM_PREFIX}{prefix}{FIM_SUFFIX}{fim_suffix_final}{FIM_MIDDLE}{middle}"
    
    return {
        'prefix': prefix,
        'middle': middle,
        'suffix': fim_suffix_final,
        'ctx_before': ctx_before,
        'ctx_after': ctx_after,
        'fim_string': fim_string,
        'line_idx': line_idx,
        'split_pos': split_pos,
        'total_lines': len(lines),
        'corrupted': should_corrupt,
        'incomplete': should_truncate,
    }

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def display_fim(result: dict, metadata: dict, index: int, total: int):
    os.system('clear')
    print(bold("FIM Visualizer") + gray(f"  [{index+1}/{total}]  n=next p=prev r=regenerate q=quit"))
    print("=" * 70)
    
    if metadata:
        info_parts = []
        if 'lang' in metadata:
            info_parts.append(f"Language: {cyan(metadata['lang'])}")
        if 'path' in metadata:
            path = metadata['path']
            if len(path) > 50:
                path = "..." + path[-47:]
            info_parts.append(f"File: {gray(path)}")
        if info_parts:
            print(" | ".join(info_parts))
    
    print(gray(f"Line {result['line_idx']+1}/{result['total_lines']}, split at char {result['split_pos']}"))
    if result.get('corrupted'):
        print(red("⚠ CORRUPTED") + gray(" - Model must fix the typo/error"))
    if result.get('incomplete'):
        print(yellow("✂ INCOMPLETE") + gray(" - Suffix truncated (code still being written)"))
    print("=" * 70)
    
    print(f"\n{bold('CURSOR CONTEXT')} {gray('(~10 lines around cursor - Mamba sees this first)')}")
    print("-" * 40)
    print(cyan(result['ctx_before']) + yellow('[CURSOR]') + cyan(result['ctx_after'][:200] + '...' if len(result['ctx_after']) > 200 else result['ctx_after']))
    
    print(f"\n{bold('PREFIX')} {gray('(full context before)')}")
    print("-" * 40)
    prefix_display = result['prefix']
    if len(prefix_display) > 500:
        prefix_display = "..." + prefix_display[-497:]
    print(red(prefix_display))
    
    print(f"\n{bold('>>> EXPECTED <<<')} {gray('(model should predict this)')}")
    print("-" * 40)
    print(purple(result['middle']))
    
    print(f"\n{bold('SUFFIX')} {gray('(full context after)')}")
    print("-" * 40)
    suffix_display = result['suffix']
    if len(suffix_display) > 500:
        suffix_display = suffix_display[:497] + "..."
    print(green(suffix_display))
    
    print("\n" + "=" * 70)
    
    print(f"\n{bold('FIM STRING')} {gray('(what model sees)')}")
    print("-" * 40)
    fim = result['fim_string']
    fim = fim.replace(CURSOR_CTX, yellow(CURSOR_CTX))
    fim = fim.replace(CURSOR, yellow(CURSOR))
    fim = fim.replace(AFTER_CURSOR, yellow(AFTER_CURSOR))
    fim = fim.replace(FIM_PREFIX, yellow(FIM_PREFIX))
    fim = fim.replace(FIM_SUFFIX, yellow(FIM_SUFFIX))
    fim = fim.replace(FIM_MIDDLE, yellow(FIM_MIDDLE))
    print(fim)
    print()

def run_interactive(dataset_iter):
    history: list[tuple[dict, dict, str]] = []
    current_idx = -1
    current_content: str | None = None
    
    def fetch_next_content() -> tuple[str, dict] | None:
        skipped = 0
        while True:
            try:
                example = next(dataset_iter)
            except StopIteration:
                print(yellow("\nDataset exhausted"))
                return None
            
            content = example.get("content", "")
            lang = example.get("lang", "unknown").lower()
            
            if lang in BANNED_LANGUAGES:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"\rSkipped: {skipped}", end='', flush=True)
                continue
            
            if len(content.strip()) < 50:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"\rSkipped: {skipped}", end='', flush=True)
                continue
            
            if content.count('\n') < 3:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"\rSkipped: {skipped}", end='', flush=True)
                continue
            
            if skipped > 0:
                print(f"\rSkipped: {skipped} samples")
            
            metadata = {
                'lang': lang,
                'path': example.get("path", "unknown"),
            }
            return content, metadata
        
        return None
    
    def sample_fim(content: str) -> dict | None:
        for _ in range(10):
            result = apply_fim(content)
            if result:
                return result
        return None
    
    def next_sample():
        nonlocal current_idx, current_content, history
        
        fetched = fetch_next_content()
        if not fetched:
            return False
        
        content, metadata = fetched
        current_content = content
        result = sample_fim(content)
        if not result:
            return False
        
        history.append((result, metadata, content))
        current_idx = len(history) - 1
        return True
    
    def regenerate():
        nonlocal history, current_idx
        
        if current_idx < 0 or current_idx >= len(history):
            return False
        
        _, metadata, content = history[current_idx]
        result = sample_fim(content)
        if not result:
            return False
        
        history[current_idx] = (result, metadata, content)
        return True
    
    if not next_sample():
        print("Failed to load initial sample")
        return
    
    result, metadata, _ = history[current_idx]
    display_fim(result, metadata, current_idx, len(history))
    
    while True:
        ch = getch()
        
        if ch in ('q', '\x03'):
            print("\nBye!")
            break
        
        elif ch == 'n':
            if current_idx < len(history) - 1:
                current_idx += 1
            else:
                if not next_sample():
                    continue
            result, metadata, _ = history[current_idx]
            display_fim(result, metadata, current_idx, len(history))
        
        elif ch == 'p':
            if current_idx > 0:
                current_idx -= 1
                result, metadata, _ = history[current_idx]
                display_fim(result, metadata, current_idx, len(history))
        
        elif ch == 'r':
            if regenerate():
                result, metadata, _ = history[current_idx]
                display_fim(result, metadata, current_idx, len(history))

def main():
    hf_token = os.environ.get("HUGGINGFACE", os.environ.get("HF_TOKEN"))
    if not hf_token:
        print(yellow("Warning: No HF_TOKEN found."))
    
    print(gray("Loading The Stack dataset (streaming)..."))
    
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        split="train",
        streaming=True,
        token=hf_token
    )
    
    dataset = dataset.shuffle(seed=random.randint(0, 10000), buffer_size=10000)
    dataset_iter = iter(dataset)
    
    run_interactive(dataset_iter)

if __name__ == "__main__":
    main()
