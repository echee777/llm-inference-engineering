#!/bin/bash
# Download the official Qwen2.5-3B-Instruct-GPTQ-Int4 checkpoint.
#
# This is published by the Qwen team directly:
#   https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
#
# Prerequisites:
#   pip install huggingface_hub
#
# Usage:
#   bash download_gptq_int4.sh
#
# Then load in vLLM:
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
#       --quantization gptq \
#       --dtype half \
#       --max-model-len 2048

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"

echo "Downloading ${MODEL_ID}..."
echo "This will cache to ~/.cache/huggingface/hub/ (~2GB)"

# Option 1: huggingface-cli (preferred)
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download "${MODEL_ID}"
    echo "Done. Use --model ${MODEL_ID} in vLLM (it will find the cached files)."
    exit 0
fi

# Option 2: python fallback
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL_ID}')
print(f'Downloaded to: {path}')
print(f'Use --model ${MODEL_ID} in vLLM.')
"
