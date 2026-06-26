# NVIDIA-SMI Output Guide

```
Starting session with SessionId: terraformer-5bq2d26cu7echnp9g62ep6znc8
sh-5.2$ nvidia-smi
Wed Feb 18 21:59:21 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
| N/A   24C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```


## What `nvidia-smi` Shows

`nvidia-smi` provides a snapshot view of GPU state at a point in time.

------------------------------------------------------------------------

## Header Section

-   **NVIDIA-SMI Version** -- Version of the CLI utility.
-   **Driver Version** -- Installed NVIDIA driver version.
-   **CUDA Version** -- Maximum CUDA version supported by the driver
    (not necessarily your PyTorch build version).

------------------------------------------------------------------------

## Main Table Fields

### GPU

Index of the GPU in the system.

### Name

Model of the GPU (e.g., Tesla T4, A100).

### Persistence-M

Whether persistence mode is enabled (keeps driver loaded to reduce
startup latency).

### Bus-Id

PCIe address of the GPU (useful for multi-GPU systems).

### Disp.A

Whether the GPU is attached to a display (Off for headless servers).

### Volatile Uncorr. ECC

Count of uncorrectable ECC memory errors.

------------------------------------------------------------------------

## Performance & Power

### Fan

Fan speed (may be N/A for passively cooled GPUs).

### Temp

GPU core temperature (°C).

### Perf (P-State)

Performance state: - P0 = Maximum performance - P8 = Idle low-power
state

### Pwr:Usage/Cap

Current power draw vs. maximum power limit (watts).

------------------------------------------------------------------------

## Memory

### Memory-Usage

Used GPU memory / total GPU memory.

------------------------------------------------------------------------

## Utilization

### GPU-Util

Percentage of time the GPU was executing compute workloads.

### Compute M.

Compute mode (Default, Exclusive, etc.).

### MIG M.

Indicates whether Multi-Instance GPU (MIG) is enabled (Ampere+ GPUs
only).

------------------------------------------------------------------------

## Processes Section

Lists active processes using GPU memory or compute.

------------------------------------------------------------------------

## Summary

`nvidia-smi` gives a static snapshot of: - Power state - Temperature -
Memory usage - Compute utilization - Running processes
