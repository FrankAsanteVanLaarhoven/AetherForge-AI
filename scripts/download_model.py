"""
Download meta-llama/Meta-Llama-3-8B from Hugging Face.
Requires: HF account with accepted Meta Llama 3 license + HF token.

Run once:
    huggingface-cli login
Then:
    conda run -n ml-torch python download_model.py
"""

from huggingface_hub import snapshot_download
import os

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
LOCAL_DIR = os.path.expanduser("~/.cache/huggingface/hub/llama3-8b-base")

print(f"Downloading {MODEL_ID} -> {LOCAL_DIR}")
print("This is ~16GB of weights (bfloat16). 4-bit quantization happens at load time.")

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=["*.pt", "original/*"],  # skip redundant original weights
)

print(f"\nDone. Model saved to: {path}")
