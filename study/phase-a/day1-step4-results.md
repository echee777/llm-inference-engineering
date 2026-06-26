# Conclusion

CHATGPT:

1) Low TFLOPS ≠ memory-bound

Low TFLOPS can happen because of:
- not enough parallel tiles to occupy all SMs
- kernel launch / dispatch overhead dominating
- inability to hide latency (too few warps)
- power/clock ramp effects
- algorithm/kernel choice overhead (esp. first runs)

None of those require DRAM bandwidth saturation.

2) Your telemetry contradicts “memory-bound”

- You observed in nvtop: mem% ~ ~10% even at small batch
- SM/GPU util changes dramatically with longer runs/bigger batch
- power only hits cap when fully loaded
- If it were bandwidth-bound, you’d expect memory controller utilization to climb a lot (often tens of percent to near 100%), while compute can’t rise.
- So your experiment demonstrates underutilization at small batch, not bandwidth saturation.

CLAUDE:

Got it — but the analysis in my response doesn't change. The T4 has different numbers (65 TFLOPS peak, 320 GB/s bandwidth, ridge point ~203 FLOPs/byte) but the same conclusions hold:

Step 4 benchmark proves underutilization at small batch, not bandwidth saturation
The memory-bound argument for decode comes from arithmetic intensity math (1 FLOP/byte vs ~203 FLOPs/byte ridge point on T4)
The actual empirical proof comes from Day 2 Nsight profiling

# Benchmark Results

⏺ Benchmark ran successfully. Here are the results:
  batch=   1: 0.0 TFLOPS ( 0% of T4 peak)
  batch=   4: 0.4 TFLOPS ( 1% of T4 peak)
  batch=  16: 3.5 TFLOPS ( 5% of T4 peak)
  batch=  64: 12.0 TFLOPS (19% of T4 peak)
  batch= 256: 19.4 TFLOPS (30% of T4 peak)
  batch=1024: 23.0 TFLOPS (35% of T4 peak)

  The T4 peaks at 23 TFLOPS FP16 (~35% of the theoretical 65 TFLOPS peak) at batch=1024. This is typical — the theoretical peak assumes perfectly saturated tensor cores, while real matrix-multiply throughput is
  memory-bandwidth limited at small batch sizes and can't hit 100% even at large batches due to overhead.

Note: Further investigations with rounds=5000 instead of 200 showed that the Tesla T4 was power-limited and could only reach 23 TFLOPs with batchsize=4096 


# CODE with inline explanations
```
import torch, time  # Import PyTorch for GPU ops and time for performance timing

# Create a 4096x4096 weight matrix on the GPU in FP16
W = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# Try different batch sizes to see how utilization changes
for batch in [1, 4, 16, 64, 256, 1024]:

    # Create input tensor of shape (batch, 4096) on GPU
    x = torch.randn(batch, 4096, device='cuda', dtype=torch.float16)

    torch.cuda.synchronize()  
    # Ensure all previous GPU work is finished before timing.
    # CUDA is asynchronous — without this, timing would be inaccurate.

    start = time.perf_counter()  
    # Start high-resolution wall-clock timer.

	# Runs the matmul 200 times to make the measurement long enough to be stable (reduces noise).
    for _ in range(200):
        y = x @ W.T  
        # Matrix multiply:
        # (batch × 4096) @ (4096 × 4096)
        # Produces (batch × 4096)
        # Each multiply costs ≈ 2 * batch * 4096 * 4096 FLOPs.

    torch.cuda.synchronize()  
    # Wait until all 200 matmuls finish before stopping timer.
    # Without this, we'd only time kernel launch overhead.

    elapsed = time.perf_counter() - start  
    # Total time (seconds) for 200 matrix multiplies.

    # Compute achieved TFLOPS:
    # FLOPs per matmul = 2 * batch * 4096 * 4096
    # We ran 200 matmuls
    # Divide by elapsed time to get FLOPs/sec
    # Divide by 1e12 to convert FLOPs/sec → TFLOPs/sec
    tflops = (2 * batch * 4096 * 4096 * 200) / (elapsed * 1e12)

    print(
        f"batch={batch:4d}: "
        f"{tflops:.1f} TFLOPS "
        f"({tflops/312*100:.0f}% of A100 peak)"
    )
    # Prints:
    # - batch size (width 4, right-aligned)
    # - achieved TFLOPS (1 decimal)
    # - percentage of theoretical A100 FP16 peak (312 TFLOPS)
```