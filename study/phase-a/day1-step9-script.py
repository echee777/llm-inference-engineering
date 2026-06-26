import torch, time

print('\n\n# GPU-GPU')
for size_mb in [1, 10, 100, 1000]:
    n = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(n, device='cuda')
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    bw = (2 * size_mb * 100) / (elapsed * 1024)  # GB/s (read + write)
    print(f"{size_mb}MB: {bw:.1f} GB/s")

print('\n\n# GPU-CPU')
for size_mb in [1, 10, 100, 1000]:
    n = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(n, device='cuda')
    b = torch.empty(n)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    bw = (2 * size_mb * 100) / (elapsed * 1024)  # GB/s (read + write)
    print(f"{size_mb}MB: {bw:.1f} GB/s")

print('\n\n# CPU-GPU')
for size_mb in [1, 10, 100, 1000]:
    n = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(n)
    b = torch.empty(n, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    bw = (2 * size_mb * 100) / (elapsed * 1024)  # GB/s (read + write)
    print(f"{size_mb}MB: {bw:.1f} GB/s")

    

import torch, time
# Import PyTorch (CUDA tensors) and time for performance measurement


for size_mb in [1, 10, 100, 1000]:
    # We will test different tensor sizes to see how bandwidth scales.
    # size_mb is the size of the tensor in megabytes.

    n = size_mb * 1024 * 1024 // 4
    # Number of float32 elements needed to make size_mb MB.
    #
    # float32 = 4 bytes per element
    # size_mb * 1024 * 1024 converts MB → bytes
    # Divide by 4 to get number of float32 elements

    a = torch.randn(n, device='cuda')
    # Allocate tensor 'a' in GPU memory (GDDR6 on T4).
    # torch.randn initializes it with random values.
    #
    # This allocates size_mb MB in DRAM.

    b = torch.empty_like(a)
    # Allocate another GPU tensor of same size.
    # Also size_mb MB in DRAM.
    #
    # No data initialized — just memory reserved.

    torch.cuda.synchronize()
    # Ensure any prior GPU work completes before timing.
    # CUDA is asynchronous — without this, timing would be inaccurate.

    start = time.perf_counter()
    # Start high-resolution CPU timer.
    # We measure elapsed wall time for 100 copies.
 
    for _ in range(100):
        b.copy_(a)
        # This is the key memory-transfer operation.
        #
        # What happens under the hood:
        #   - Read size_mb MB from DRAM (tensor a)
        #   - Write size_mb MB to DRAM (tensor b)
        #
        # So each iteration moves:
        #   2 × size_mb MB of data
        #
        # There is essentially no math here.
        # This kernel is bandwidth-bound.

    torch.cuda.synchronize()
    # Wait until all 100 copy kernels complete.
    # Ensures we measure actual execution time.

    elapsed = time.perf_counter() - start
    # Total time in seconds for 100 read+write passes.

    bw = (2 * size_mb * 100) / (elapsed * 1024)
    # Compute bandwidth in GB/s.
    #
    # 2 × size_mb → read + write per iteration
    # × 100 → 100 iterations
    #
    # Divide by elapsed time → MB per second
    # Divide by 1024 → convert MB/s to GB/s (approx GiB/s)

    print(f"{size_mb}MB: {bw:.1f} GB/s")
    # Print measured sustained memory bandwidth.
    
