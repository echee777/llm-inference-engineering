# Captured output

```
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     13     24      -      0      0      0      0      0      0    405    300
    0     31     26      -     89     37      0      0      0      0   5000   1590
    0     32     26      -      0      0      0      0      0      0   5000   1590
    0     25     25      -      0      0      0      0      0      0   5000    375
    0     24     25      -      0      0      0      0      0      0   5000    375
    0     24     25      -      0      0      0      0      0      0   5000    300
    0     24     25      -      0      0      0      0      0      0   5000    300
    0     24     25      -      0      0      0      0      0      0   5000    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
    0     13     25      -      0      0      0      0      0      0    405    300
```



# SM caveat
"The percentage of time during the sampling interval that any SM had at least one warp executing instructions."

Important Distinction

- Case A:
  - All SMs fully saturated
  - Tensor cores maxed
  - Perfect occupancy

- Case B:
  - Only one warp active per SM
  - Low arithmetic intensity

- Both could show:
  - sm% ≈ 100%





# NVIDIA-SMI DMON Output Guide

## What `nvidia-smi dmon` Shows

`nvidia-smi dmon` (device monitor) provides real-time, per-second GPU
telemetry.

------------------------------------------------------------------------

## Column Breakdown

### gpu / Idx

GPU index number.

### pwr (W)

Current power draw in watts.

### gtemp (C)

GPU core temperature (°C).

### mtemp (C)

Memory temperature (if supported).

------------------------------------------------------------------------

## Utilization Metrics

### sm (%)

Streaming Multiprocessor utilization. - Percentage of time SMs were
actively executing instructions. - Time-based metric (not peak FLOP
efficiency).

### mem (%)

Memory controller utilization. - Indicates bandwidth usage. - High
mem% + low sm% often indicates memory bottleneck.

------------------------------------------------------------------------

## Media Engines

### enc

Video encoder utilization.

### dec

Video decoder utilization.

### jpg

JPEG hardware engine utilization.

### ofa

Optical Flow Accelerator utilization.

------------------------------------------------------------------------

## Clock Speeds

### mclk (MHz)

Memory clock frequency. - Scales up under load. - Drops during idle
(power saving).

### pclk (MHz)

Processor (SM core) clock frequency. - Ramps up during compute
workloads. - Drops in idle states.

------------------------------------------------------------------------

## Key Interpretation Patterns

-   **High sm%, low mem%** → Likely compute-bound workload.
-   **High mem%, lower sm%** → Likely memory-bandwidth-bound workload.
-   **Clocks increase under load** → Dynamic Voltage and Frequency
    Scaling (DVFS).
-   **Power near cap** → Potential power-limited performance.

------------------------------------------------------------------------

## Summary

`nvidia-smi dmon` is useful for observing:

-   Compute vs. memory bottlenecks
-   Dynamic clock scaling
-   Power draw behavior
-   Real-time workload patterns
