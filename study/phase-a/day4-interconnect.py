import torch, time  # Import PyTorch (for tensors + CUDA ops) and time (for timing measurements)

# Test multiple transfer sizes in megabytes
for size_mb in [1, 10, 100, 500, 1000]:

    # Compute number of float32 elements needed to equal size_mb megabytes.
    # 1 MB = 1024 * 1024 bytes
    # float32 = 4 bytes per element
    # So total elements = (MB * bytes_per_MB) / 4
    n = size_mb * 1024 * 1024 // 4  # number of float32 elements

    # Allocate a random tensor of size n on the CPU
    cpu_tensor = torch.randn(n)

    # ----------------------------
    # CPU → GPU transfer (Host to Device, H2D)
    # ----------------------------

    # Ensure all previous CUDA operations are complete before starting timing.
    # CUDA is asynchronous — without this, timing would be incorrect.
    torch.cuda.synchronize()

    # Record start time (high-resolution timer)
    start = time.perf_counter()

    # Perform the transfer 50 times to smooth out timing noise.
    # .to('cuda') copies tensor from CPU memory → GPU memory over PCIe/NVLink.
    for _ in range(50):
        gpu_tensor = cpu_tensor.to('cuda')

    # Wait for all CUDA operations to finish before stopping timer.
    torch.cuda.synchronize()

    # Compute total elapsed time
    elapsed = time.perf_counter() - start

    # Bandwidth calculation:
    # We moved size_mb MB per iteration × 50 iterations
    # Convert MB to GB by dividing by 1024
    # Bandwidth = total_GB / time_seconds
    h2d_bw = (size_mb * 50) / (elapsed * 1024)  # GB/s

    # ----------------------------
    # GPU → CPU transfer (Device to Host, D2H)
    # ----------------------------

    # Allocate fresh random tensor directly on GPU
    gpu_tensor = torch.randn(n, device='cuda')

    # Synchronize again before timing
    torch.cuda.synchronize()

    # Start timing GPU → CPU transfers
    start = time.perf_counter()

    # Copy GPU tensor back to CPU 50 times
    # .to('cpu') triggers device → host copy
    for _ in range(50):
        cpu_tensor = gpu_tensor.to('cpu')

    # Ensure all transfers are finished
    torch.cuda.synchronize()

    # Compute elapsed time
    elapsed = time.perf_counter() - start

    # Same bandwidth formula for device-to-host
    d2h_bw = (size_mb * 50) / (elapsed * 1024)  # GB/s

    # Print formatted result:
    # size in MB
    # H2D bandwidth
    # D2H bandwidth
    print(f"{size_mb:4d} MB: H2D={h2d_bw:.1f} GB/s, D2H={d2h_bw:.1f} GB/s")