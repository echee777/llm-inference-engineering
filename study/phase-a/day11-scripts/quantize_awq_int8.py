"""
Quantize Qwen2.5-3B-Instruct to INT8-AWQ using llm-compressor.

Prerequisites (run on your GPU instance):
    pip install llmcompressor datasets

Usage:
    python quantize_awq_int8.py

Output:
    ./qwen2.5-3b-int8-awq/   (local directory with quantized model)

Then load in vLLM:
    python -m vllm.entrypoints.openai.api_server \
        --model ./qwen2.5-3b-int8-awq \
        --quantization awq \
        --dtype half \
        --max-model-len 2048
"""

import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./qwen2.5-3b-int8-awq"

# Calibration dataset config.
# ultrachat_200k is a standard choice for calibration.
# 256 samples is the recommended starting point.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# AWQ recipe:
#   scheme:  W8A16 = INT8 weights, FP16 activations (weight-only quantization).
#            W4A16_ASYM would be INT4 asymmetric.
#   targets: Apply to all Linear layers.
#   ignore:  Skip lm_head (final projection to vocab) to preserve output quality.
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W8A16",
        targets=["Linear"],
    ),
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded in {time.time() - t0:.1f}s")

print(f"Loading calibration dataset ({NUM_CALIBRATION_SAMPLES} samples)...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)

print(f"Quantizing to INT8-AWQ (W8A16)...")
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"Quantized in {time.time() - t0:.1f}s")

# Save quantized model.
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Load with: LLM('{OUTPUT_DIR}', quantization='awq')")
