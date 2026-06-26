import torch, time

W = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
for batch in [1, 4, 16, 64, 256, 1024]:
    x = torch.randn(batch, 4096, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        y = x @ W.T
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = (2 * batch * 4096 * 4096 * 200) / (elapsed * 1e12)
    print(f"batch={batch:4d}: {tflops:.1f} TFLOPS ({tflops/65*100:.0f}% of T4 peak)")


    
