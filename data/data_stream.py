from datasets import load_dataset

# safe_licenses = ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"]
# test_langs = ["Dockerfile"]

# Stream the full Stack v2 (900B tokens, 600+ languages)
dataset = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",
    split="train",
    streaming=True,
)

def code_to_fim(code):
    return code

for i in dataset:
    print(i["content"])
    break