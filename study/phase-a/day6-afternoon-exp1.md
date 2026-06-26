# Experiment 1 (max-num-seq sweep)
# Shows how server-side concurrency constrains RPS (Decode throughput)

- python3 day6-exp1.py
- Runs 16 concurrent client requests
- While limiting server side (vllm) concurrency
	- max-num-seq limits concurrency at vllm level (even though client has concurrency=16)

## Encountered error
```
ERROR: Cannot use FA version 2 ... FA2 is only supported on devices with compute capability >= 8
→ Falls back to FlashInfer
→ FlashInfer also fails during warmup on CC 7.5
→ Engine core initialization failed
```

### Claude said to pin to an older version of vllm
pip install vllm==0.6.6.post1 --break-system-packages

## Encountered error

Clean error this time — vLLM 0.6.6 is running (XFormers backend selected correctly), but TinyLlama's config defaults to bfloat16 and the T4 doesn't support it. One flag fixes it:

### set dtype half

python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1 \
  --dtype half \
  --port 8000 \
  --disable-log-requests
  
## Working

((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ nvidia-smi
Thu Mar  5 07:06:32 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0             67W /   70W |   12847MiB /  15360MiB |     89%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            6283      C   ...sm-user/venv-vllm/bin/python3      12834MiB |
+-----------------------------------------------------------------------------------------+


# Increased requests to 66 for enough repetition
# Increase concurrency 33 so that client concurrency 
# exceeds vLLM server concurrency 16
  
((venv-vllm) ) [ssm-user@ip-10-99-0-199 ~]$ python3 day6-exp1.py

================================================================================
Day 6 — Experiment 1: --max-num-seqs Parameter Sensitivity
Model:       TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU Mem:     0.85
Sweep:       [1, 4, 8, 16, 32]
Requests:    66 @ concurrency=33
================================================================================

────────────────────────────────────────────────────────────
  RUN: --max-num-seqs=1
────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-seqs 1
  Command: /home/ssm-user/venv-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 1 --dtype half --port 8000 --disable-log-requests
  Server ready after 1s
  Warming up (2 requests)...
  Sending 66 requests at concurrency=33...
  ✓ Throughput:      84.3 tok/s
    TTFT p50:      21.668s
    TTFT p99:      36.572s
    Latency p50:   22.874s
    Latency p99:   36.999s
    Total tokens:    5099
    Wall time:       60.5s
    Success:     66/66
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --max-num-seqs=4
────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-seqs 4
  Command: /home/ssm-user/venv-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 4 --dtype half --port 8000 --disable-log-requests
  Server ready after 1s
  Warming up (2 requests)...
  Sending 66 requests at concurrency=33...
  ✓ Throughput:      84.5 tok/s
    TTFT p50:      30.687s
    TTFT p99:      44.436s
    Latency p50:   33.081s
    Latency p99:   44.921s
    Total tokens:    5488
    Wall time:       65.0s
    Success:     66/66
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --max-num-seqs=8
────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-seqs 8
  Command: /home/ssm-user/venv-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 8 --dtype half --port 8000 --disable-log-requests
  Server ready after 1s
  Warming up (2 requests)...
  Sending 66 requests at concurrency=33...
  ✓ Throughput:      84.1 tok/s
    TTFT p50:      26.748s
    TTFT p99:      37.195s
    Latency p50:   27.649s
    Latency p99:   37.735s
    Total tokens:    5726
    Wall time:       68.1s
    Success:     66/66
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --max-num-seqs=16
────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-seqs 16
  Command: /home/ssm-user/venv-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 16 --dtype half --port 8000 --disable-log-requests
  Server ready after 1s
  Warming up (2 requests)...
  Sending 66 requests at concurrency=33...
  ✓ Throughput:      85.7 tok/s
    TTFT p50:      32.713s
    TTFT p99:      47.927s
    Latency p50:   33.912s
    Latency p99:   48.857s
    Total tokens:    6441
    Wall time:       75.2s
    Success:     66/66
  Stopping vLLM...
  Waiting 5s for GPU memory release...

────────────────────────────────────────────────────────────
  RUN: --max-num-seqs=32
────────────────────────────────────────────────────────────

  Starting vLLM: --max-num-seqs 32
  Command: /home/ssm-user/venv-vllm/bin/python3 -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 32 --dtype half --port 8000 --disable-log-requests
  Server ready after 1s
  Warming up (2 requests)...
  Sending 66 requests at concurrency=33...
  ✓ Throughput:      83.9 tok/s
    TTFT p50:      28.167s
    TTFT p99:      34.520s
    Latency p50:   28.385s
    Latency p99:   35.789s
    Total tokens:    4959
    Wall time:       59.1s
    Success:     66/66
  Stopping vLLM...
  Waiting 5s for GPU memory release...

# Bad results.  Throughput bounded at 84

================================================================================
EXPERIMENT 1 RESULTS: --max-num-seqs Parameter Sensitivity
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompt: ~512 tokens | Max completion: 256 tokens
Requests per run: 66 | Concurrency: 33
================================================================================
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|   max-num-seqs |   Throughput |   TTFT p50 |   TTFT p99 |   Latency p50 |   Latency p99 |   Success |
|                |      (tok/s) |        (s) |        (s) |           (s) |           (s) |      Rate |
+================+==============+============+============+===============+===============+===========+
|              1 |         84.3 |     21.668 |     36.572 |        22.874 |        36.999 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|              4 |         84.5 |     30.687 |     44.436 |        33.081 |        44.921 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|              8 |         84.1 |     26.748 |     37.195 |        27.649 |        37.735 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|             16 |         85.7 |     32.713 |     47.927 |        33.912 |        48.857 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|             32 |         83.9 |     28.167 |     34.52  |        28.385 |        35.789 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+


# Reran next morning, no code changes
# Could not repeat throughput boundedness
# Maybe previous night was overheated


================================================================================
EXPERIMENT 1 RESULTS: --max-num-seqs Parameter Sensitivity
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompt: ~512 tokens | Max completion: 256 tokens
Requests per run: 66 | Concurrency: 33
================================================================================
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|   max-num-seqs |   Throughput |   TTFT p50 |   TTFT p99 |   Latency p50 |   Latency p99 |   Success |
|                |      (tok/s) |        (s) |        (s) |           (s) |           (s) |      Rate |
+================+==============+============+============+===============+===============+===========+
|              1 |         84.7 |     21.646 |     36.342 |        22.837 |        36.763 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|              4 |        233.8 |      7.994 |     10.61  |         8.772 |        12.728 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|              8 |        323.7 |      4.182 |      5.583 |         5.377 |        10.255 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|             16 |        383.7 |      2.134 |      3.219 |         3.29  |         9.532 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+
|             32 |        488.4 |      0.359 |      2.631 |         3.622 |         9.402 |     66/66 |
+----------------+--------------+------------+------------+---------------+---------------+-----------+


# While running the above, watched clock, temp, cpu/mem sat

```
gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk     fb   bar1   ccpm
Idx      W      C      C      %      %      %      %      %      %    MHz    MHz     MB     MB     MB
    0     13     32      -      0      0      0      0      0      0    405    300      0      2      0
    0     13     32      -      0      0      0      0      0      0    405    300      0      2      0
    0     13     31      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     31      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     31      -      0      0      0      0      0      0    405    300      0      2      0
    0     10     31      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     30      -      0      0      0      0      0      0    405    300      0      2      0
    0     10     30      -      0      0      0      0      0      0    405    300      0      2      0
    0     10     30      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     30      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     30      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     30      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     29      -      0      0      0      0      0      0    405    300      0      2      0
    0      9     29      -      0      0      0      0      0      0    405    300      0      2      0
    0     10     29      -      0      0      0      0      0      0    405    300      3      3      0
    0     33     30      -     32      1      0      0      0      0   5000   1590   2277      5      0
    0     34     30      -      1      0      0      0      0      0   5000   1590   2329      5      0
    0     33     31      -      0      0      0      0      0      0   5000   1590  12713      5      0
    0     33     31      -      0      0      0      0      0      0   5000   1590  12719      5      0
    0     33     31      -      0      0      0      0      0      0   5000   1590  12723      5      0
    0     33     31      -      0      0      0      0      0      0   5000   1590  12747      5      0
    0     35     31      -      0      0      0      0      0      0   5000   1590  12753      5      0
    0     33     31      -      0      0      0      0      0      0   5000   1590  12755      5      0
    0     68     33      -     92     94      0      0      0      0   5000   1305  12855      5      0
    0     68     33      -     92     94      0      0      0      0   5000   1290  12855      5      0
    0     69     34      -     92     94      0      0      0      0   5000   1290  12855      5      0
    0     68     34      -     97     49      0      0      0      0   5000    840  13509      5      0
    0     56     34      -     94     59      0      0      0      0   5000   1140  13509      5      0
    0     76     35      -     93     61      0      0      0      0   5000    840  13509      5      0
    0     71     35      -     91     75      0      0      0      0   5000   1335  13509      5      0
    0     58     36      -     96     58      0      0      0      0   5000    795  13509      5      0
    0     64     36      -     93     61      0      0      0      0   5000   1350  13509      5      0
    0     75     36      -     93     70      0      0      0      0   5000   1185  13509      5      0
    0     59     36      -     94     65      0      0      0      0   5000   1320  13509      5      0
    0     71     37      -     91     85      0      0      0      0   5000   1335  13509      5      0
    0     69     37      -     91     88      0      0      0      0   5000   1440  13515      5      0
    0     33     36      -     85     87      0      0      0      0   5000   1590  13515      5      0
    0     33     35      -      0      0      0      0      0      0   5000   1590  13515      5      0
    0     33     35      -     22      4      0      0      0      0   5000   1590      0      2      0
    0     33     35      -      0      0      0      0      0      0   5000   1590      0      2      0
```

# Analysis

Raw results saved to day6_exp1_results.json
Done. Review the table above and record observations in your notes.

Key questions to answer:
  • At what --max-num-seqs does throughput plateau on your T4?
  • How does TTFT p99 change as you allow more concurrent sequences?
  • Did any runs hit errors (rejected/preempted requests)?
  • What's the sweet spot for your T4 + TinyLlama setup?
  

## Results

The experiment measured how changing --max-num-seqs affects throughput and latency for TinyLlama running on a T4 GPU.

Throughput increased rapidly as concurrency increased from 1 to 16 sequences, showing that the GPU was initially underutilized.

Throughput gains slowed significantly between 16 and 32 sequences, indicating the system was approaching a hardware bottleneck.

The maximum observed throughput was about 500 tokens/sec at --max-num-seqs=32.

Time-to-first-token improved dramatically as batching increased because the GPU processed multiple prompts simultaneously during the prefill phase.

Tail latency improved initially with batching but stopped improving much beyond around 16 sequences.

All runs succeeded with no request failures, indicating the system handled the concurrency safely.

GPU memory usage (~13 GB of 16 GB) was not the limiting factor, so VRAM capacity was not the bottleneck.

The flattening of the throughput curve suggests the workload is becoming limited by GPU memory bandwidth rather than compute capacity.

Transformer decoding workloads repeatedly read the KV cache from DRAM, which increases memory traffic as concurrency increases.

Once DRAM bandwidth approaches saturation, adding more concurrent sequences no longer increases throughput.

Nsight Compute profiling can confirm this by showing high DRAM utilization relative to SM compute utilization.

For this model and hardware, the practical sweet spot appears to be around --max-num-seqs=32, where throughput is near maximum without additional gains from higher concurrenc  
  