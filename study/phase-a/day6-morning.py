# Avoid FlashInfer because of dependency on nvcc
#
# vLLM
#  └── FlashInfer attention backend
#       └── JIT compile CUDA kernel
#            └── requires nvcc
#
# To avoid, set --attention-config.backend TRITON_ATTN
# 
# Orthogonally, set gptq_marlin for better quantization performance than gptq
# 
# Can ignore the vllm.entrypoints.chat_utils.ChatTemplateResolutionError
# because this older model only supports /v1/completions NOT /v1/chat/completions
# the Chat API supports role=system content=xxx, role=user content=yyy
#
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-GPTQ \
  --quantization gptq_marlin \
  --max-model-len 2048 \
  --dtype float16 \
  --attention-config.backend TRITON_ATTN

###########
# CLIENT
###########

curl -s http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "TheBloke/Llama-2-7B-GPTQ",
    "prompt": "Answer with one word.\n\nThe capital of France is",
    "max_tokens": 3,
    "temperature": 0,
    "stop": ["\n"]
  }' | jq
  
###########
# RESPONSE
###########
{
  "id": "cmpl-a073173ea36a98f6",
  "object": "text_completion",
  "created": 1772687936,
  "model": "TheBloke/Llama-2-7B-GPTQ",
  "choices": [
    {
      "index": 0,
      "text": " Paris.",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": "\n",
      "token_ids": null,
      "prompt_logprobs": null,
      "prompt_token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 13,
    "total_tokens": 16,
    "completion_tokens": 3,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}


###########
# TinyLlama
###########

```
python -m vllm.entrypoints.openai.api_server   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0    --max-model-len 2048   --dtype float16   --attention-config.backend TRITON_ATTN
```