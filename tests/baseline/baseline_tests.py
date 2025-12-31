from tests.model_engines import OpenRouterAdapter  

ad = OpenRouterAdapter("qwen/qwen3-coder")

import os
print(os.getcwd())


def format_fim(code):
    find = code.find("<cursor>")
    if find == -1:
        return code
    return f"<PRE>{code[:find]}<SUF>{code[find+7:]} <MID>"

def parse_file(file):
    fd = open(file, "r")
    code = fd.read().split("\n")
    
    target = code[0][2:]

    fim = format_fim("\n".join(code[1:]))
    return(target, fim) 

filecontent = parse_file("./tests/baseline/categories/sql/sql2.py")


print("===EXPECTED==")
print(filecontent[0])
print("=====GOT=====")
print(ad.request_completion(filecontent[1]))