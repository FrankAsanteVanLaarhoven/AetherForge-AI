"""
Generate dummy multimodal dataset for testing --mode multimodal fine-tuning.

Creates:
  multimodal_example/images/   — 20 synthetic PNG images (solid colour + text label)
  multimodal_example/multimodal_data.jsonl — JSONL with instruction/response/image

Usage:
    cd /path/to/AetherForge-AI
    conda run -n ml-torch python multimodal_example/generate_dummy_data.py

Then fine-tune:
    conda run -n ml-torch python scripts/finetune_qwen25_vl.py \
        --mode multimodal \
        --data multimodal_example/multimodal_data.jsonl
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent
IMAGE_DIR = OUTPUT_DIR / "images"
JSONL_PATH = OUTPUT_DIR / "multimodal_data.jsonl"

IMAGE_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    ("hospital corridor", (200, 220, 240), "Navigate carefully — pedestrians present."),
    ("operating room",    (220, 240, 220), "Sterile environment — reduce speed."),
    ("pharmacy",          (240, 230, 210), "Identify medication labels on shelves."),
    ("patient room",      (240, 220, 200), "Low noise required — patient resting."),
    ("reception desk",    (230, 210, 240), "Queue present — wait before approaching."),
    ("fire exit",         (255, 180, 180), "Emergency route — keep clear at all times."),
    ("supply room",       (210, 240, 210), "Restock items from lower shelves first."),
    ("radiology suite",   (200, 200, 240), "No metal objects — MRI zone."),
    ("cafeteria",         (255, 240, 200), "Busy at lunch — plan alternate path."),
    ("elevator lobby",    (220, 220, 220), "Wait for doors — do not block entry."),
]

QA_TEMPLATES = [
    ("What environment is shown in this image?",
     "The image shows a {label} in a hospital setting."),
    ("What action should a mobile robot take here?",
     "{advice}"),
    ("Describe the scene and any safety considerations.",
     "This is a {label}. {advice}"),
    ("What label or zone is depicted?",
     "The depicted zone is: {label}."),
    ("Is this area safe for autonomous robot navigation?",
     "Navigating the {label} requires caution. {advice}"),
]

records = []
for i, (label, colour, advice) in enumerate(CATEGORIES):
    for repeat in range(2):   # 2 examples per category = 20 total
        img_name = f"{i:02d}_{repeat}_{label.replace(' ', '_')}.png"
        img_path = IMAGE_DIR / img_name

        # Draw a solid-colour image with the label as white text
        img = Image.new("RGB", (224, 224), color=colour)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 100), label, fill=(255, 255, 255), font=font)
        img.save(img_path)

        qa = random.choice(QA_TEMPLATES)
        instruction = qa[0]
        response = qa[1].format(label=label, advice=advice)

        records.append({
            "instruction": instruction,
            "response": response,
            "image": str(img_path.resolve()),
        })

random.shuffle(records)

with open(JSONL_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Generated {len(records)} examples")
print(f"Images:  {IMAGE_DIR}")
print(f"Dataset: {JSONL_PATH}")
print("\nSample record:")
print(json.dumps(records[0], indent=2))
