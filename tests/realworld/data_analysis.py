import os
import random
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from datasets import load_dataset
import re
import dotenv

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

FIM_PREFIX = "<|pre|>"
FIM_SUFFIX = "<|suf|>"
FIM_MIDDLE = "<|mid|>"
OTHER_CTX = "<|ctx|>"
CURSOR = "<|cur|>"

MAX_SUFFIX_CHARS = 1000

def get_special_tokens():
    return [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, OTHER_CTX, CURSOR]

def format_fim_string(sample: dict) -> str:
    mode = sample.get("mode", "spm")
    ctx = sample.get("context", "")
    prefix = sample.get("prefix", "")
    suffix = sample.get("suffix", "")
    middle = sample.get("middle", "")
    
    if mode == "spm":
        return f"{OTHER_CTX}{ctx}{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"
    elif mode == "prefix_only":
        return f"{OTHER_CTX}{ctx}{FIM_SUFFIX}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"
    elif mode == "suffix_only":
        return f"{OTHER_CTX}{ctx}{FIM_SUFFIX}{suffix}{FIM_PREFIX}{FIM_MIDDLE}{middle}"
    return f"{OTHER_CTX}{ctx}{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"

def format_inference_prompt(prefix: str, suffix: str, context: str = "") -> str:
    suffix = suffix[:MAX_SUFFIX_CHARS] if len(suffix) > MAX_SUFFIX_CHARS else suffix
    return f"{OTHER_CTX}{context}{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"

def get_other_context(content, cursor_line=None):
    """Extract imports, class/function signatures with docstrings, and module-level assignments"""
    tree = parser.parse(bytes(content, "utf8"))
    content_lines = content.split("\n")
    total_lines = len(content_lines)
    if cursor_line is None:
        cursor_line = total_lines
    
    definitions = []
    
    def get_docstring(node):
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.child_count > 0 else None
                if expr and expr.type == "string":
                    doc = expr.text.decode('utf8')
                    if len(doc) > 100:
                        doc = doc[:100] + '..."""'
                    return doc
        return None
    
    def get_return_type(node):
        ret = node.child_by_field_name("return_type")
        if ret:
            return " -> " + ret.text.decode('utf8')
        return ""
    
    def calc_proximity_weight(node):
        node_line = node.start_point[0]
        distance = abs(cursor_line - node_line)
        return max(0, 1.0 - (distance / total_lines))
    
    def extract_definitions(node, depth=0):
        if depth > 50:
            return
        weight = calc_proximity_weight(node)
        
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            superclasses = node.child_by_field_name("superclasses")
            if name_node:
                class_sig = f"class {name_node.text.decode('utf8')}"
                if superclasses:
                    class_sig += superclasses.text.decode('utf8')
                docstring = get_docstring(node)
                if docstring:
                    class_sig += f":\n    {docstring}"
                definitions.append((weight, class_sig))
                
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            if name_node:
                params = params_node.text.decode('utf8') if params_node else "()"
                ret_type = get_return_type(node)
                func_sig = f"def {name_node.text.decode('utf8')}{params}{ret_type}"
                docstring = get_docstring(node)
                if docstring:
                    func_sig += f":\n    {docstring}"
                definitions.append((weight, func_sig))
                
        elif node.type in ("import_statement", "import_from_statement"):
            definitions.append((1.0, node.text.decode('utf8')))
            
        elif node.type == "expression_statement" and depth == 0:
            child = node.children[0] if node.child_count > 0 else None
            if child and child.type == "assignment":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if left and left.type == "identifier":
                    var_name = left.text.decode('utf8')
                    if not var_name.startswith("_"):
                        type_hint = child.child_by_field_name("type")
                        if type_hint:
                            definitions.append((weight, f"{var_name}: {type_hint.text.decode('utf8')}"))
                        elif right:
                            right_text = right.text.decode('utf8')
                            if len(right_text) > 50:
                                right_text = right_text[:50] + "..."
                            definitions.append((weight, f"{var_name} = {right_text}"))
        
        for child in node.children:
            extract_definitions(child, depth + 1)
    
    extract_definitions(tree.root_node)
    definitions.sort(key=lambda x: -x[0])
    return "\n".join([d[1] for d in definitions])


class PythonFIMStackDataset:
    def __init__(self):
        # auth me into hf
        dotenv.load_dotenv()
        hf_token = os.environ.get("HF_TOKEN", "uhhh no token for you")
        if not hf_token:
            print("Warning: No HF_TOKEN found.")
        self.dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            token=hf_token
        )

    def is_valid(self, content):
        lines = content.split("\n")
        num_lines = len(lines)
        
        # Line count bounds
        if num_lines > 8000 or num_lines < 40:
            return False
        
        # Max line length > 1000 (minified code / data blobs)
        max_line_length = max(len(line) for line in lines)
        if max_line_length > 1000:
            return False
        
        # Avg line length > 100 (obfuscated / packed code)
        avg_line_length = len(content) / max(1, num_lines)
        if avg_line_length > 100:
            return False
        
        # Alphanumeric density < 25% (hex dumps / binary data)
        alnum_count = sum(c.isalnum() for c in content)
        if (alnum_count / len(content)) < 0.25:
            return False
        
        # Assignment ratio check (< 0.05 means likely just data/imports)
        eq_count = content.count('=')
        if (eq_count / num_lines) < 0.05:
            return False
        
        # Autogenerated patterns
        autogenerated_patterns = [
            r"<autogenerated\s*/>",
            r"generated by",
            r"auto-generated",
            r"do not edit",
            r"machine generated",
        ]
        content_lower = content.lower()
        for pattern in autogenerated_patterns:
            if re.search(pattern, content_lower):
                return False
        
        return True

    def generate_fim(self, content):
        # Define some probabilities
        scenarios = [("Normal", 0.7), ("Surrounded", 0.05), ("Start", 0.15), ("End", 0.1)]
        # roll the dice to see what it is
        scenario = random.choices([x[0] for x in scenarios], weights=[x[1] for x in scenarios], k=1)[0]
        # We will do the repitition thing in a selective fine tune
        modifiers = [("Normal", 0.7), ("Cutoff", 0.3)]
        modifier = random.choices([x[0] for x in modifiers], weights=[x[1] for x in modifiers], k=1)[0]
        
        def find_bracket_region(line):
            """Find a region inside (), [], or {} - returns (start, end) or None"""
            brackets = {'(': ')', '[': ']', '{': '}'}
            for i, char in enumerate(line):
                if char in brackets:
                    close_char = brackets[char]
                    depth = 1
                    for j in range(i + 1, len(line)):
                        if line[j] == char:
                            depth += 1
                        elif line[j] == close_char:
                            depth -= 1
                            if depth == 0:
                                if j - i > 2:
                                    return (i + 1, j)
                                break
            return None
        
        def generate_split(codedata):
            """Finds a spot to cut the file and creates the content left and right of the split"""
            lines = codedata.split("\n")
            line_num = random.randint(0, len(lines) - 1)
            line = lines[line_num]
            # pick a random spot in the line.
            while 0 >= len(line) - 1:
                line_num = random.randint(0, len(lines) - 1)
                line = lines[line_num]
            
            spot = random.randint(0, len(line) - 1)
            if scenario == "Start":
                spot = 0
            elif scenario == "End":
                spot = len(line)-1
            elif scenario == "Surrounded":
                bracket_region = find_bracket_region(line)
                if bracket_region:
                    start, end = bracket_region
                    region_len = end - start
                    # weight toward start of bracket region
                    spot = start + int(random.triangular(0, region_len, 0))
                    # pick end point, also weighted toward start (shorter completions)
                    remaining = end - spot
                    if remaining > 1:
                        split_offset = int(random.triangular(1, remaining, 1))
                        line_piece = line[spot:spot + split_offset]
                        pre = "\n".join(lines[0:line_num]) + "\n" + line[:spot]
                        post = line[spot + split_offset:] + "\n" + "\n".join(lines[line_num+1:])
                        return (pre, line_piece, post, line_num)
                # fallback to normal if no bracket found
                scenario_fallback = "Normal"
            
            pre = "\n".join(lines[0:line_num]) + "\n" + line[:spot]
            line_piece = line[spot:]
            post = "\n" + "\n".join(lines[line_num+1:])
            return (pre, line_piece, post, line_num)
        
        def get_cursor_context(split):
            """should get the block around the cursor like 200 chars maybe"""
            pre, post = split[0], split[2]
            cursor_pre = pre[-200:]
            cursor_post = post[:200]
            return (cursor_pre, cursor_post)
        
        pre, line, post, cursor_line = generate_split(content)
        cursor_pre, cursor_post = get_cursor_context((pre, line, post))
        # eval cutoff prob
        if modifier == "Cutoff":
            post = ""
            cursor_post = ""
        
        context = get_other_context(pre, cursor_line)
        
        modes = [("spm", 0.70), ("prefix_only", 0.29), ("suffix_only", 0.01)]
        mode = random.choices([m[0] for m in modes], weights=[m[1] for m in modes], k=1)[0]
        
        if len(post) > MAX_SUFFIX_CHARS:
            post = post[:MAX_SUFFIX_CHARS]
            
        result = {
            "scenario": scenario,
            "modifier": modifier,
            "mode": mode,
            "cursor": cursor_pre + cursor_post,
            "context": context,
            "prefix": pre,
            "suffix": post,
            "middle": line,
        }
        
        return result
        

        
        
        
        
        
    def __iter__(self):
        for item in self.dataset:
            content = item["content"]
            if not self.is_valid(content):
                continue
            
            content_lines = content.split('\n')
            content_lines = [line for line in content_lines if line.strip() != ""]

            yield self.generate_fim("\n".join(content_lines)),