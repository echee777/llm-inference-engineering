"""
Day 28: Kill observation experiment.
Start streaming requests, kill -15 vLLM, record client-side error mode.
"""
import asyncio
import aiohttp
import time
import signal
import subprocess
import os

VLLM_BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


async def stream_request(session, request_id, results):
    """Send a streaming request and record what happens."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Write a detailed essay about the history of computing. Request {request_id}."}],
        "max_tokens": 256,
        "stream": True,
    }

    tokens_received = 0
    error_mode = None
    t0 = time.monotonic()

    try:
        async with session.post(
            f"{VLLM_BASE}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                error_mode = f"http_{resp.status}"
                results.append((request_id, tokens_received, error_mode, time.monotonic() - t0))
                return

            async for line in resp.content:
                decoded = line.decode('utf-8').strip()
                if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                    tokens_received += 1

    except aiohttp.ClientConnectionError as e:
        error_mode = f"connection_error: {str(e)[:100]}"
    except aiohttp.ServerDisconnectedError as e:
        error_mode = f"server_disconnected: {str(e)[:100]}"
    except asyncio.TimeoutError:
        error_mode = "timeout"
    except Exception as e:
        error_mode = f"other: {type(e).__name__}: {str(e)[:100]}"

    elapsed = time.monotonic() - t0
    if error_mode is None:
        error_mode = "completed_normally"

    results.append((request_id, tokens_received, error_mode, elapsed))


async def main():
    print("=== Kill Observation Experiment ===")
    print(f"Sending 3 concurrent streaming requests, then kill -15 vLLM after ~5 tokens stream")
    print()

    # Get vLLM PID
    result = subprocess.run(["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')
    if not pids or pids == ['']:
        print("ERROR: No vLLM process found")
        return
    vllm_pid = int(pids[0])
    print(f"vLLM PID: {vllm_pid}")

    results = []

    async with aiohttp.ClientSession() as session:
        # Start 3 streaming requests
        tasks = [
            asyncio.create_task(stream_request(session, i, results))
            for i in range(3)
        ]

        # Wait a bit for tokens to start streaming
        await asyncio.sleep(3)

        # Kill vLLM
        print(f"Sending SIGTERM to PID {vllm_pid}...")
        os.kill(vllm_pid, signal.SIGTERM)
        print("SIGTERM sent. Waiting for client responses...")

        # Wait for all requests to finish (or error out)
        await asyncio.gather(*tasks, return_exceptions=True)

    print()
    print("=== RESULTS ===")
    print(f"{'ID':<4} {'Tokens':<8} {'Error Mode':<50} {'Duration (s)':<10}")
    print("-" * 80)
    for req_id, tokens, error, duration in results:
        print(f"{req_id:<4} {tokens:<8} {error:<50} {duration:.3f}")

    print()
    # Summary
    error_modes = set(r[2] for r in results)
    print(f"Distinct error modes observed: {error_modes}")
    print()
    if any("disconnect" in r[2].lower() or "connection" in r[2].lower() or "reset" in r[2].lower() for r in results):
        print("CLIENT SEES: Connection reset / server disconnected (will trigger retry)")
    elif any("completed" in r[2] for r in results):
        print("CLIENT SEES: Clean completion (may NOT trigger retry)")
    else:
        print(f"CLIENT SEES: {error_modes}")


if __name__ == "__main__":
    asyncio.run(main())
