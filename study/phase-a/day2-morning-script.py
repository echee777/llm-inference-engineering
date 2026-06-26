# Hugging Face Transformers: provides pretrained models + tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch: CUDA runtime + profiler + inference utilities
import torch

# Use a ~3B-class model that fits comfortably on a T4 in FP16.
model_name = "microsoft/phi-2"

# Load the tokenizer that converts text -> token IDs for this model.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Some models/tokenizers (including Phi-2 variants) may not define a pad token.
# Generation often needs a pad token ID, especially when we pass attention masks / padding.
# A common safe choice is to reuse the EOS token as PAD.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model onto the GPU in FP16.
# - torch_dtype=torch.float16 saves VRAM and matches tensor core friendly dtype on T4.
# - device_map="cuda" places model weights on GPU.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
)

# Set inference/eval mode (turns off dropout etc.).
model.eval()

# --------------------------------------------
# Build a LONG prompt so "prefill" is actually heavy.
# --------------------------------------------
# If the prompt is too short, prefill is tiny and your profiler will mostly show CPU plumbing.
# Repeating a short sentence many times makes the prompt length large (hundreds to ~1000+ tokens).
base = "Explain the roofline model in simple terms. "
prompt = base * 400  # Increase/decrease if you want more/less prefill work.

# Tokenize the prompt into tensors:
# - input_ids: token IDs the model consumes
# - attention_mask: 1s where tokens are real, 0s where padding exists
# padding=True ensures the tokenizer returns attention_mask consistently.
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1900)

# Move inputs to GPU (must match the model device).
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

# --------------------------------------------
# Warmup (IMPORTANT)
# --------------------------------------------
# First run often includes one-time overhead:
# - CUDA context init
# - kernel caching
# - allocator settling
# Warmup reduces noise so your profiling reflects real steady-state behavior.
with torch.inference_mode():  # Disables autograd; faster + lower memory
    _ = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,                 # minimal decode; mostly prefill
        pad_token_id=tokenizer.eos_token_id,  # explicit pad token for stable behavior
        do_sample=False,                  # deterministic generation, avoids sampling overhead
    )

# Ensure the GPU finishes all work before we start profiling.
torch.cuda.synchronize()

# --------------------------------------------
# Helper: run a profiling session and print top CUDA ops
# --------------------------------------------
def run_profile(label: str, max_new_tokens: int):
    # torch.profiler records CPU and GPU activities during the block below.
    # This gives you an operator-level "what ops dominate time" view.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,   # measure CPU overhead
            torch.profiler.ProfilerActivity.CUDA,  # measure GPU kernel time
        ],
        record_shapes=True,  # record tensor shapes to help interpret ops
	with_stack=False,    # stack traces add overhead; off is fine for this exercise
    ) as prof:
        if label == 'DECODE-HEAVY':
            # inference_mode disables autograd and reduces overhead/memory.
            with torch.inference_mode():
                _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
        else:
	    with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)


    # Ensure GPU work is complete before printing profiler results.
    torch.cuda.synchronize()

    # Print a table of the top ops sorted by total CUDA time.
    # This is the main Step-3 deliverable: compare top ops prefill vs decode.
    print(f"\n=== {label} (max_new_tokens={max_new_tokens}) ===")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # If you want the table to always show CUDA columns, use row_limit=50 and sort by self_cuda_time_total (sometimes cleaner):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
# --------------------------------------------
# Profile 1: "Prefill-dominant"
# --------------------------------------------
# max_new_tokens=1 means:
# - Do the full prefill pass for the long prompt
# - Then decode only 1 token
# So most of the measured time should correspond to prefill.
run_profile("PREFILL-DOMINANT", max_new_tokens=1)

# --------------------------------------------
# Profile 2: "Decode-heavy"
# --------------------------------------------
# max_new_tokens=50 means:
# - Prefill once (same long prompt)
# - Then run 50 sequential decode steps
# This will surface many repeated small kernels typical of decode.
run_profile("DECODE-HEAVY", max_new_tokens=50)






