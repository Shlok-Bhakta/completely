# FIM (Fill-in-Middle) Format

## Overview

Single-line completion FIM format with cursor context for Mamba state space models. The model sees ~10 lines of local context around the cursor first (for Mamba's state), then receives the full file context to predict the completion.

## Special Tokens

```
«CTX»      - Start of cursor context (local ~10 lines)
«CURSOR»   - Cursor position marker
«/CTX»     - End of cursor context
«PREFIX»   - Full code before cursor
«SUFFIX»   - Full code after cursor
«MIDDLE»   - What model should generate (completes to end of line)
```

## Format

```
«CTX»{ctx_before}«CURSOR»{ctx_after}«/CTX»«PREFIX»{prefix}«SUFFIX»{suffix}«MIDDLE»{middle}
```

## Key Design Decisions

### 1. Guillemet Tokens (« »)

Changed from `<tag>` format to `«TAG»` to avoid conflicts with HTML/XML/JSX code in training data.

### 2. Cursor Context for Mamba

Mamba processes sequences linearly and maintains hidden state. By showing ~10 lines of local context first, the model's state is "primed" with the immediate surroundings before seeing the full file. This helps Mamba focus on the relevant context.

- `ctx_before`: Up to 10 lines before cursor + partial line before cursor
- `ctx_after`: Up to 10 lines after cursor (or empty in incomplete mode)

### 3. Cursor Position Weighting

Split points are **weighted toward the start of lines**:
- Position 0 (start of line) is most common
- Each additional character is less likely
- Uses linear decay: `weight = 1.0 - (position / (length + 1)) * 0.5`

This matches real usage where users often start typing new lines.

### 4. End-of-Line = Predict Whole Next Line

When cursor is at end of a line, the middle target is the **entire next line**.
This trains the model for the common "press enter, get suggestion" flow.

### 5. Incomplete Suffix Mode (35% of samples)

Simulates "code still being written" scenarios:
- `ctx_after` is empty (nothing after cursor in context)
- `suffix` is empty (no code after in FIM)
- Trains model to complete with minimal context

This teaches the model to work when users are actively typing and haven't written the rest of the file yet.

### 6. Language Filtering

Training excludes data formats to focus on actual code:
- Banned: CSV, JSON, YAML, XML, SVG, SQL, Markdown, Jupyter notebooks, Turtle, API Blueprint
- Keeps all programming languages (including niche ones like Perl, Elixir, etc.)

### 7. Newline Handling

Middle completion always ends with a newline character, ensuring the model learns proper line boundaries.

## Training Modes

### Normal Mode (65%)
```
«CTX»{~10 lines before}«CURSOR»{~10 lines after}«/CTX»«PREFIX»{full prefix}«SUFFIX»{full suffix}«MIDDLE»{completion}
```

### Incomplete Mode (35%)
```
«CTX»{~10 lines before}«CURSOR»«/CTX»«PREFIX»{full prefix}«SUFFIX»«MIDDLE»{completion}
```
Empty suffix simulates actively-being-written code.

## Examples

### Example 1: Normal completion mid-line

**Original:**
```python
def calculate_sum(x, y):
    result = x + y
    return result
```

**FIM (cursor after "result = "):**
```
«CTX»def calculate_sum(x, y):
    result = «CURSOR»x + y
    return result«/CTX»«PREFIX»def calculate_sum(x, y):
    result = «SUFFIX»x + y
    return result«MIDDLE»x + y
```

### Example 2: Incomplete mode (code being written)

**Original:**
```python
def foo():
    x = 1
```

**FIM (cursor after "x = ", incomplete mode):**
```
«CTX»def foo():
    x = «CURSOR»«/CTX»«PREFIX»def foo():
    x = «SUFFIX»«MIDDLE»1
```
Note: No context or suffix after cursor - simulates typing mid-file.

### Example 3: End of line completion

**Original:**
```javascript
function greet(name) {
    console.log("Hello " + name)
}
```

**FIM (cursor at end of line 1):**
```
«CTX»function greet(name) {«CURSOR»
    console.log("Hello " + name)
}«/CTX»«PREFIX»function greet(name) {
«SUFFIX»
    console.log("Hello " + name)
}«MIDDLE»    console.log("Hello " + name)
```

## Training Data Generation

For each code file:
1. Filter out banned languages (data formats)
2. Skip files < 50 chars or < 3 lines
3. Generate FIM samples:
   - Pick random line and weighted split point
   - Decide mode: 35% incomplete, 65% normal
   - Build cursor context (~10 lines around cursor)
   - Build full prefix/suffix
   - If incomplete: empty ctx_after and suffix
4. Always append newline to middle completion

## Inference

```
«CTX»{local_context_before}«CURSOR»{local_context_after}«/CTX»«PREFIX»{code_before_cursor}«SUFFIX»{code_after_cursor}«MIDDLE»
```

Generate until newline or max tokens. Output is the line completion.
