import os
import gc
import pandas as pd
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/medgemma-1.5-4b-it"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
CSV_OUTPUT = os.path.join(BASE_DIR, "vlm_results.csv")

MAX_IMAGES = 10
MAX_NEW_TOKENS = 60

os.makedirs(REPORT_FOLDER, exist_ok=True)

# ============================================================
# 4BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MedGemma (4bit)...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

print("Model loaded successfully")

# ============================================================
# LOAD IMAGES
# ============================================================

image_paths = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
][:MAX_IMAGES]

print(f"Processing {len(image_paths)} images")

# ============================================================
# PROMPT
# ============================================================

def build_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are an expert radiologist.\n"
                        "Analyze this chest X-ray.\n\n"
                        "Return:\n\n"
                        "Findings:\n"
                        "- ...\n\n"
                        "Abnormalities:\n"
                        "- ...\n\n"
                        "Impression:\n"
                        "- Pneumonia likely or unlikely"
                    ),
                },
            ],
        }
    ]

# ============================================================
# GENERATE REPORTS
# ============================================================

results = []

for i, img_path in enumerate(image_paths):

    print(f"\n[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}")

    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))

        messages = build_messages()

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # ⭐ DO NOT MOVE TO CUDA
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        report = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        if len(report) < 10:
            report = "No clear findings generated."

        report_file = os.path.join(
            REPORT_FOLDER,
            os.path.basename(img_path) + ".txt"
        )

        with open(report_file, "w") as f:
            f.write(report)

        results.append({
            "image": img_path,
            "report": report
        })

        print("✔ Done")

        del inputs, outputs
        gc.collect()

    except Exception as e:
        print("❌ Error:", e)

# ============================================================
# SAVE CSV
# ============================================================

if results:
    pd.DataFrame(results).to_csv(CSV_OUTPUT, index=False)

print("\nAll reports generated!")

