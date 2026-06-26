import subprocess
import time
import requests
import signal
import os

def start_vllm(model, port=8000, **kwargs):
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--max-model-len", "2048",
        "--port", str(port),
    ]
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for health endpoint
    for i in range(90):
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                print(f"  Server ready after {i+1}s")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)
    
    proc.kill()
    raise RuntimeError("vLLM failed to start")

def stop_vllm(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)  # let GPU memory release

# Experiment 2
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

for gpu_util in [0.5, 0.7, 0.85, 0.95]:
    print(f"\n{'='*50}")
    print(f"gpu-memory-utilization = {gpu_util}")
    print(f"{'='*50}")
    
    proc = start_vllm(MODEL, gpu_memory_utilization=gpu_util)
    
    # Run your benchmarks here
    for conc in [2, 4, 8, 12, 16, 20]:
        result = asyncio.run(run_benchmark(
            "http://localhost:8000", MODEL,
            prompt, 256, concurrency=conc, num_requests=conc
        ))
    
    stop_vllm(proc)