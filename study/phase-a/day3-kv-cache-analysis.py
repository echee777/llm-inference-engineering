def kv_cache_analysis(
    gpu_memory_gb,
    model_size_gb,
    num_layers,
    num_kv_heads,
    head_dim,
    dtype_bytes=2,  # FP16
    overhead_pct=0.10
):
    """Calculate max concurrent requests for various sequence lengths."""
    available_gb = (gpu_memory_gb - model_size_gb) * (1 - overhead_pct)
    available_bytes = available_gb * 1e9
    
    kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
    
    print(f"GPU: {gpu_memory_gb} GB | Model: {model_size_gb} GB | Available for KV: {available_gb:.1f} GB")
    print(f"KV per token: {kv_per_token:,} bytes ({kv_per_token/1024:.1f} KB)")
    print(f"KV per token (all layers): {kv_per_token:,} bytes ({kv_per_token/1e6:.2f} MB)")
    print()
    
    for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
        kv_per_request = kv_per_token * seq_len
        max_concurrent = int(available_bytes // kv_per_request)
        print(f"  seq_len={seq_len:6d}: {kv_per_request/1e9:.2f} GB/req → max {max_concurrent} concurrent")

# Your T4
# print("=== Tesla T4 (16 GB) ===")
# kv_cache_analysis(16, 14, 32, 32, 128)
#
# # Interview reference: A100
# print("\n=== A100-40GB ===")
# kv_cache_analysis(40, 14, 32, 32, 128)


print("=== T4: FP16 KV Cache ===")
kv_cache_analysis(16, 14, 32, 32, 128, dtype_bytes=2)

print("\n=== T4: INT8 KV Cache ===")
kv_cache_analysis(16, 14, 32, 32, 128, dtype_bytes=1)

print("\n=== A100: FP16 vs INT8 ===")
kv_cache_analysis(40, 14, 32, 32, 128, dtype_bytes=2)
kv_cache_analysis(40, 14, 32, 32, 128, dtype_bytes=1)