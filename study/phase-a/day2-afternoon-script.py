# Hugging Face Transformers: provides pretrained models + tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch: CUDA runtime + profiler + inference utilities
import torch


# ----------------------------
# Model choice
# ----------------------------
# Use a ~3B-class model that fits comfortably on a T4 in FP16.
# NOTE: Phi-2 is ~2.7B. It should fit on a 16GB T4 with FP16 weights.
model_name = "microsoft/phi-2"

# Load the tokenizer that converts text -> token IDs for this model.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Some models/tokenizers may not define a pad token.
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


# ----------------------------
# Tokenization
# ----------------------------
# Tokenize the prompt into tensors:
# - input_ids: token IDs the model consumes
# - attention_mask: 1s where tokens are real, 0s where padding exists
#
# IMPORTANT: truncation/max_length ensures we don't exceed the model context and skew results.
# For Day 2, we mainly want: "prefill is a big pass" vs "decode is many small steps".
MAX_PROMPT_TOKENS = 1024
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_PROMPT_TOKENS,
)

# Move inputs to GPU (must match the model device).
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

# Print actual prompt tokens so you know what you’re profiling.
print(f"[info] prompt_tokens={input_ids.shape[1]} (capped at {MAX_PROMPT_TOKENS})")


# --------------------------------------------
# Warmup (IMPORTANT)
# --------------------------------------------
# First run often includes one-time overhead:
# - CUDA context init
# - kernel caching
# - allocator settling
# Warmup reduces noise so your profiling reflects real steady-state behavior.
#
# IMPORTANT FIX: Warm up the same execution path you will profile (generate()).
with torch.inference_mode():  # Disables autograd; faster + lower memory
    _ = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,                     # minimal decode; mostly prefill
        pad_token_id=tokenizer.eos_token_id,  # explicit pad token for stable behavior
        do_sample=False,                      # deterministic generation; avoids sampling overhead
        use_cache=True,                       # ensure KV cache path is active
    )

# Ensure the GPU finishes all work before we start profiling.
torch.cuda.synchronize()


# --------------------------------------------
# Helper: generate within NVTX ranges (for nsys/ncu)
# --------------------------------------------
def nvtx_generate(label: str, max_new_tokens: int):
    '''
    Run model.generate() inside an NVTX range so Nsight Systems/Compute can label the region.
    This is the core Step-4 (nsys/ncu) primitive.
    '''
    # NVTX ranges appear in Nsight Systems if you run with: --trace=nvtx
    torch.cuda.nvtx.range_push(label)
    try:
        with torch.inference_mode():
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )
    finally:
        torch.cuda.nvtx.range_pop()

    # Ensure GPU work is complete before returning (important for clean measurement boundaries).
    torch.cuda.synchronize()


# --------------------------------------------
# Step 3: torch.profiler run (operator-level breakdown)
# --------------------------------------------
def run_torch_profiler(label: str, max_new_tokens: int):
    '''
    torch.profiler records CPU and GPU activities during the block below.
    This gives you an operator-level "what ops dominate time" view (Day 2 Step 3).
    '''
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,   # measure CPU overhead
            torch.profiler.ProfilerActivity.CUDA,  # measure GPU kernel time
        ],
        record_shapes=True,  # record tensor shapes to help interpret ops
        with_stack=False,    # stack traces add overhead; off is fine for this exercise
    ) as prof:
        # IMPORTANT FIX: Use generate() in BOTH modes so you compare identical code paths.
        # For "prefill-dominant", max_new_tokens=1 ensures prefill dominates the run.
        nvtx_generate(label=f"NVTX::{label}", max_new_tokens=max_new_tokens)

    # Print a table of the top ops sorted by self CUDA time.
    print(f"\n=== {label} (max_new_tokens={max_new_tokens}) ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))


# --------------------------------------------
# Main entrypoint
# --------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prefill", "decode"], help="Which phase to run/profile.")
    parser.add_argument(
        "--tool",
        choices=["prof", "nsight"],
        default="prof",
        help=(
            "prof: use torch.profiler (Step 3). "
            "nsight: clean run with NVTX only (Step 4 nsys/ncu)."
        ),
    )
    args = parser.parse_args()

    # Choose generation length.
    # - prefill: do the long prompt pass + decode 1 token (prefill dominates)
    # - decode:  do the long prompt pass + decode 50 tokens (decode-heavy)
    if args.mode == "prefill":
        label = "PREFILL-DOMINANT"
        new_tokens = 1
    else:
        label = "DECODE-HEAVY"
        new_tokens = 50

    if args.tool == "prof":
        # Step 3 deliverable: torch.profiler operator breakdown.
        run_torch_profiler(label=label, max_new_tokens=new_tokens)
    else:
        # Step 4 deliverable: clean run for Nsight (no torch.profiler overhead).
        #
        # IMPORTANT:
        # Nsight Systems' *stats* will summarize the whole process unless you restrict capture.
        # You can capture *only* this NVTX region by adding:
        #   --capture-range=nvtx --capture-range-end=stop
        #
        # We'll also call torch.cuda.profiler.start/stop so you *optionally* can use:
        #   --capture-range=cudaProfilerApi --capture-range-end=stop
        #
        # (Pick one capture method; NVTX capture is usually simplest.)
        print(
            "[hint] For cleaner Nsight stats (exclude model load/tokenize), run:
"
            "  nsys profile --trace=cuda,nvtx,osrt --stats=true "
            "--capture-range=nvtx --capture-range-end=stop --force-overwrite true "
            "-o day2 /opt/pytorch/bin/python3 day2-afternoon.py "
            f"{args.mode} --tool nsight"
        )

        # Start CUDA profiler range (optional for Nsight capture-range=cudaProfilerApi).
        try:
            torch.cuda.profiler.start()
            _profiler_started = True
        except Exception:
            _profiler_started = False

        # Add an outer NVTX range to make it obvious on the nsys timeline (and usable for capture-range=nvtx).
        torch.cuda.nvtx.range_push("DAY2_STEP4_NSIGHT_RUN")
        try:
            nvtx_generate(label=label, max_new_tokens=new_tokens)
        finally:
            torch.cuda.nvtx.range_pop()
            if _profiler_started:
                try:
                    torch.cuda.profiler.stop()
                except Exception:
                    pass

