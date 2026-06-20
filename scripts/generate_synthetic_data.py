"""
scripts/generate_synthetic_data.py
Downloads a sample of Alpaca instruction-following data from HuggingFace
and saves it as data/synthetic_data.jsonl for use with distillation/finetune.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "synthetic_data.jsonl"

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

try:
    from datasets import load_dataset
    print("Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
    records = []
    for row in ds:
        if row.get("output") and row.get("instruction"):
            inp = row.get("input", "").strip()
            instr = row["instruction"].strip()
            if inp:
                instr = f"{instr}\n\n{inp}"
            records.append({"instruction": instr, "response": row["output"].strip()})
    random.shuffle(records)
    records = records[:N_SAMPLES]
except Exception as e:
    print(f"HuggingFace download failed ({e}), generating local synthetic data …")
    # Fallback: simple template-based synthetic pairs
    TEMPLATES = [
        ("Explain {topic} in simple terms.", "Here is a simple explanation of {topic}: {topic} is a fundamental concept that involves ..."),
        ("What is {topic}?", "{topic} is defined as a concept or entity that has specific properties and characteristics ..."),
        ("Write a Python function to {task}.", "Here is a Python function to {task}:\n\ndef solve():\n    # Implementation\n    pass"),
        ("Summarize the following: {topic}.", "Summary: The key points about {topic} are as follows ..."),
        ("How do you {task}?", "To {task}, follow these steps:\n1. First, prepare the necessary resources\n2. Then apply the main procedure\n3. Finally, verify the result"),
    ]
    TOPICS = [
        "machine learning", "neural networks", "Python programming", "data structures",
        "algorithms", "linear algebra", "statistics", "natural language processing",
        "computer vision", "reinforcement learning", "optimization", "transformers",
        "attention mechanisms", "gradient descent", "backpropagation", "regularization",
    ]
    TASKS = [
        "sort a list", "reverse a string", "compute the factorial", "find the maximum element",
        "check if a number is prime", "implement binary search", "merge two sorted lists",
        "find all permutations", "count word frequencies", "parse JSON data",
    ]
    records = []
    for i in range(N_SAMPLES):
        tmpl, resp_tmpl = random.choice(TEMPLATES)
        topic = random.choice(TOPICS)
        task  = random.choice(TASKS)
        instr = tmpl.format(topic=topic, task=task)
        resp  = resp_tmpl.format(topic=topic, task=task)
        records.append({"instruction": instr, "response": resp})

with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Wrote {len(records)} samples → {OUT}")
