# Analysis

So the correct summary is:

- Decode launches scale with number of generated tokens.
  - So basically weights need to be streamed across the model topology for a forward pass for each token in decode right?
  - Each token causes the model weights to be streamed from HBM again.
  - Many kernel launches correspond to a forward pass eval of the weights per the model topology where in a GPU kernel (function) corresponds to a model operation (tensor op, parallel math, compute tile)
  - Serving batch size dramatically improves decode throughput (can reuse weights across multiple tokens in batch)

- Prefill launches scale mostly with number of layers.
  - Prefill does NOT divide decode launch count by number of prompt tokens.


## Prefill vs Decode Kernel Characterization (microsoft/Phi-2 on T4)

| Phase | Representative Kernel | Achieved Occupancy | SM Active % | Memory Throughput (% peak) | Compute Throughput (% peak) | Primary Stall Reason |
|-------|-----------------------|--------------------|-------------|-----------------------------|-----------------------------|----------------------|
| **Prefill (long prompt)** | `fmha_cutlass` / large `s1688gemm` | High (~70–90%) | High | ~25–40% | ~80–85% | Compute / TensorCore saturation |
| **Decode (single token)** | `cublasGemvParamsEx` (GEMV) | Moderate | ~100% active cycles | **~96%** | **~58%** (~7% of FP32 peak) | DRAM bandwidth |

---

## Key Contrast

- **Prefill:** Compute-bound, high arithmetic intensity, large GEMMs.
- **Decode:** Memory-bound, low arithmetic intensity, GEMV-heavy.  

# Prefill Kernel Roofline Analysis (T4 / sm75)

------------------------------------------------------------------------

## 1️⃣ Dominant GEMM Is Compute-Saturated

The primary prefill GEMM kernel\
`turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn`\
achieves:

-   **Compute Throughput:** 83.87% of peak\
-   **SM Active:** \~98.8%\
-   **DRAM Throughput:** 24.59% of peak\
-   **Memory Throughput:** 38.58% of peak

**Conclusion:**\
The dominant GEMM is **compute-bound** and operating near Tensor Core
saturation.\
Performance improvements would require algorithmic changes, kernel
improvements, or fusion --- not simply increasing batch size.

------------------------------------------------------------------------

## 2️⃣ Attention (FMHA) Is Not Compute- or DRAM-Bound

The fused attention kernel\
`fmha_cutlassF_f16_aligned_32x128_rf_sm75`

achieves:

-   **Compute Throughput:** 27.82% of peak\
-   **SM Active:** \~95%\
-   **DRAM Throughput:** 6.50% of peak\
-   **Memory Throughput:** 41.71%\
-   **L1/TEX Throughput:** 83.40%

**Conclusion:**\
Attention is **not compute-saturated** and **not DRAM
bandwidth-bound**.\
Instead, it is **latency-limited and cache-heavy**, relying heavily on
L1/TEX paths.

------------------------------------------------------------------------

## 3️⃣ Prefill Is GEMM-Dominated

Nsight Systems previously showed \~60% of GPU time spent in GEMMs.

Roofline confirms:

-   GEMMs are compute-heavy and near peak efficiency.
-   Attention is secondary and under-utilized.

**Conclusion:**\
For prefill, overall performance is governed primarily by large dense
matrix multiplications.

------------------------------------------------------------------------

## 4️⃣ Latency vs Throughput Distinction

Although the FMHA kernel shows high SM active cycles (\~95%),\
its compute throughput is low (\~28%).

This indicates:

-   Dependency chains\
-   Warp synchronization\
-   Instruction mix inefficiency\
-   Limited instruction-level parallelism

**Conclusion:**\
High SM activity does not imply high compute utilization.\
Attention is **latency-constrained**, not throughput-constrained.

------------------------------------------------------------------------

## 5️⃣ Architectural Insight for Inference Design

On T4 (sm75):

-   Prefill → compute-bound GEMMs dominate.\
-   Attention → latency-sensitive, cache-heavy behavior.\
-   DRAM bandwidth is not the primary limiter in this configuration.

**Implication:**\
Optimizing inference on older GPUs requires:

-   Better kernel fusion\
-   Reduced synchronization overhead\
-   Improved attention implementations\
-   Or migration to architectures with stronger attention performance
    characteristics

  

-------------------------------------------------------------------------------
DECODE
-------------------------------------------------------------------------------

# Decode Analysis

### 1️⃣ Clean Capture
Profiling window excluded model load and setup using `cudaProfilerStart/Stop`.  
Results reflect steady-state decode only.

---

### 2️⃣ High Kernel Launch Count (~53K)
Decode triggered ~53,000 `cudaLaunchKernel` calls.  
Launch count scales roughly with:

tokens × layers × ops_per_layer

This confirms decode is launch-heavy.

---

### 3️⃣ Kernel Mix = GEMV + Small Ops
Dominant kernels were matrix-vector (`gemvx`) and elementwise ops.  
This is characteristic of batch=1 decode (low arithmetic intensity).

---

### 4️⃣ Frequent CPU Synchronization
Significant `cudaStreamSynchronize` time indicates CPU waits between GPU segments.  
Execution is fragmented.

---

### 5️⃣ Minimal PCIe Traffic
Host↔Device memcpy volume ≈ negligible.  
Bottleneck is internal GPU execution and HBM weight streaming.

---

## Decode Signature

- Many small kernels  
- Launch + sync overhead sensitive  
- Memory-bandwidth influenced  
- Fundamentally sequential (tokens × layers)

# Decode output

/opt/pytorch/lib64/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████| 453/453 [00:01<00:00, 318.26it/s, Materializing param=model.layers.31.self_attn.v_proj.weight]
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-beb1.qdstrm'
[1/8] [0%                          ] day2_decode.nsys-repProcessing events...
[1/8] [========================100%] day2_decode.nsys-rep
Processing 164781 events:
[2/8] [========================100%] day2_decode.sqlite
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Style                  Range
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------  -----------------------------------
     50.0       2092519989          1  2092519989.0  2092519989.0  2092519989  2092519989          0.0  PushPop  :DAY2_STEP4_NSIGHT_RUN
     50.0       2092396017          1  2092396017.0  2092396017.0  2092396017  2092396017          0.0  PushPop  :DECODE-HEAVY
      0.0            65891          2       32945.5       32945.5       27193       38698       8135.3  PushPop  CCCL:cub::DeviceScan::InclusiveScan

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ---------------------
    100.0       3763633112        195  19300682.6  10063851.0  10059626  100125962   27389953.1  poll
      0.0           461130          2    230565.0    230565.0     38784     422346     271219.3  pthread_rwlock_wrlock
      0.0           216262          5     43252.4     51948.0     11890      76402      27671.5  ioctl
      0.0            70120          5     14024.0     14564.0      9876      19792       4085.5  pthread_cond_signal
      0.0            60590          2     30295.0     30295.0     27488      33102       3969.7  pthread_rwlock_rdlock
      0.0            13444          1     13444.0     13444.0     13444      13444          0.0  munmap
      0.0            13272          1     13272.0     13272.0     13272      13272          0.0  mmap

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)              Name
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------------
     58.1        378324514      53062      7129.9      6563.0      4723    7744669      33709.0  cudaLaunchKernel
     37.3        243212533        155   1569113.1      5423.0      3575  220447274   17696194.9  cudaStreamSynchronize
      2.1         13921180          1  13921180.0  13921180.0  13921180   13921180          0.0  cuLibraryLoadData
      1.6         10288184      53062       193.9       185.0       131      16953        138.4  cuKernelGetName
      0.6          3698560        360     10273.8      9385.0      4969      33756       4008.0  cudaMemcpyAsync
      0.3          1928305       1651      1168.0      1121.0       951      17955        665.6  cudaStreamIsCapturing_v10000
      0.0           268459          1    268459.0    268459.0    268459     268459          0.0  cudaMalloc
      0.0            17484          1     17484.0     17484.0     17484      17484          0.0  cudaDeviceSynchronize
      0.0             9049          6      1508.2      1185.5       906       2688        683.7  cuLibraryGetKernel
      0.0             8811          1      8811.0      8811.0      8811       8811          0.0  cuProfilerStart

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     42.7        731880100       7890    92760.5    55999.0     54304   1065526      97070.8  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (b…
     18.7        320096131       1568   204142.9   204126.0    202334    206687        654.6  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (b…
     10.1        172344962       3136    54956.9    55551.5     48192     62687       4159.1  void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(…
      9.0        154643438       1600    96652.1    83007.0     75680    918552      98451.5  fmha_cutlassF_f16_aligned_32x128_rf_sm75(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, c…
      6.1        104913334        160   655708.3   448652.0    288765   1825776     477866.4  turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn
      3.4         58359061         32  1823720.7  1892719.0   1378612   2086861     202721.6  turing_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_stages_32x1_tn
      2.0         34434779       6464     5327.2     3328.0      2656     98783      12301.1  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>,…
      1.5         24946725       7936     3143.5     1568.0      1312    261310      16933.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, std:…
      1.4         23640526       4800     4925.1     1472.0      1248    180606      24166.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<c10::Half, c10::Ha…
      1.3         21839577       6400     3412.4     2784.0      2368     36928       4099.6  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.8         13221498       1650     8013.0     7008.0      6208     61919       7083.0  void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::Half, float, (bool)0>(int, T2, const …
      0.6         10555976       1600     6597.5     1536.0      1344    262750      35446.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::Half, c10::Ha…
      0.6         10287929       3200     3215.0     2880.0      2272     22176       2577.8  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.5          8152963       1600     5095.6     1632.0      1440    181023      24186.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::tanh_kernel_cuda(at::TensorItera…
      0.5          8002421       1600     5001.5     1536.0      1344    180479      24184.3  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      0.5          7910421       1600     4944.0     1504.0      1312    181694      24169.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<c10::Half>…
      0.1          2229772         64    34840.2    35855.0     28800     37919       2645.5  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.1           929976         50    18599.5    18496.0     16480     27072       1393.4  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<…
      0.0           292413        100     2924.1     3264.0      1856      5440        799.7  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      0.0           258399         50     5168.0     5152.0      4768      5696        190.6  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrappe…
      0.0           256638         50     5132.8     5120.0      4736      6528        241.3  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrappe…
      0.0           254206        100     2542.1     2496.0      2144      3552        220.2  void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(…
      0.0           203806        151     1349.7     1344.0      1152      2048        106.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<bool>, std::array<ch…
      0.0           186174         50     3723.5     3712.0      3232      4320        193.3  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      0.0           171744        100     1717.4     1696.0      1472      2560        132.5  void at::native::vectorized_elementwise_kernel<(int)2, at::native::BinaryFunctor<long, long, long, …
      0.0           170880        100     1708.8     1696.0      1472      1888         77.9  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctorOnSelf_add<long>, std…
      0.0           166463         49     3397.2     3423.0      3008      3712        129.4  void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void …
      0.0           160831        100     1608.3     1600.0      1376      2400        125.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<bool, bool, bool, …
      0.0           150016        100     1500.2     1504.0      1312      1664         64.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::float16_copy_kernel_cuda(at::Ten…
      0.0           146207        100     1462.1     1472.0      1248      1632         66.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, floa…
      0.0           136448         50     2729.0     2672.0      2368      4032        237.4  void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<long, long, long, at::native…
      0.0           135808         49     2771.6     2784.0      2464      3008         96.4  void at::native::<unnamed>::indexSelectSmallIndex<c10::Half, long, unsigned int, (int)2, (int)2, (i…
      0.0           116478         50     2329.6     2304.0      1984      2688        136.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIterat…
      0.0           115869         50     2317.4     2304.0      1984      2624        132.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIterat…
      0.0           107744         49     2198.9     2208.0      1952      2368         78.1  void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(…
      0.0           102047         50     2040.9     2016.0      1760      2912        150.1  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<long>…
      0.0            91807         49     1873.6     1856.0      1696      2080         67.3  void gemmk1_kernel<int, float, (int)256, (int)5, (bool)1, (bool)0, (bool)0, (bool)0, cublasGemvTens…
      0.0            86463         51     1695.4     1696.0      1504      2464        125.8  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<long, long, bool, …
      0.0            81759         50     1635.2     1632.0      1440      2368        124.1  void at::native::vectorized_elementwise_kernel<(int)4, at::native::bitwise_not_kernel_cuda(at::Tens…
      0.0            81119         50     1622.4     1600.0      1472      2367        120.9  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctor_add<long>, std::arra…
      0.0            80510         50     1610.2     1600.0      1440      2368        123.4  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctorOnOther_add<long>, st…
      0.0            69789         52     1342.1     1344.0      1152      1984        108.6  void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<ch…
      0.0            40000         25     1600.0     1600.0      1408      1696         64.0  void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<long, long, bool, at::native…
      0.0            26944         14     1924.6     1888.0      1664      2688        240.1  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<long, long, bool, …
      0.0            22592          1    22592.0    22592.0     22592     22592          0.0  void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon…
      0.0            22368         12     1864.0     1856.0      1760      1952         45.5  void at::native::vectorized_elementwise_kernel<(int)2, at::native::BinaryFunctor<long, long, bool, …
      0.0             5568          2     2784.0     2784.0      2560      3008        316.8  void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_…
      0.0             4960          2     2480.0     2480.0      2240      2720        339.4  void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<long>, std::array<ch…
      0.0             3264          1     3264.0     3264.0      3264      3264          0.0  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>,…
      0.0             2144          2     1072.0     1072.0      1056      1088         22.6  void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<lon…
      0.0             1760          1     1760.0     1760.0      1760      1760          0.0  void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTens…
      0.0             1568          1     1568.0     1568.0      1568      1568          0.0  void at::native::unrolled_elementwise_kernel<void at::native::compare_scalar_kernel<long>(at::Tenso…
      0.0             1568          1     1568.0     1568.0      1568      1568          0.0  void at::native::vectorized_elementwise_kernel<(int)2, at::native::<unnamed>::masked_fill_kernel(at…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     64.3           277566    205    1354.0    1344.0      1120      2048         92.3  [CUDA memcpy Device-to-Device]
     35.3           152319    152    1002.1     992.0       896      2144        111.1  [CUDA memcpy Device-to-Host]
      0.3             1472      3     490.7     384.0       352       736        213.1  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      0.025    205     0.000     0.000     0.000     0.008        0.001  [CUDA memcpy Device-to-Device]
      0.000    152     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]
      0.000      3     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Host-to-Device]



# ncu (gemv)

## Profile run

/opt/nvidia/nsight-compute/2025.4.1/ncu --force  --target-processes all   --set full   --kernel-name-base demangled   --kernel-name regex:gemvx   --launch-count 1   -o roofline_decode_gemv   /opt/pytorch/bin/python3 day2-afternoon.py decode --tool nsight

## Roofline

[ssm-user@ip-10-99-0-199 ~]$ /opt/nvidia/nsight-compute/2025.4.1/ncu   --import /home/ssm-user/roofline_decode_gemv.ncu-rep   --section SpeedOfLight
[3625] python3.12@127.0.0.1
  enable_if<!T7, void>::type kernel<int, int, __half, __half, __half, float, 0, 1, 1, 0, 6, 0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const __half>, cublasGemvTensorStridedBatched<const __half>, cublasGemvTensorStridedBatched<__half>, float>>(T13) (12800, 1, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.99
    SM Frequency                    Mhz       585.01
    Elapsed Cycles                cycle       748357
    Memory Throughput                 %        96.10
    DRAM Throughput                   %        96.10
    Duration                         ms         1.28
    L1/TEX Cache Throughput           %        87.05
    L2 Cache Throughput               %        23.57
    SM Active Cycles              cycle    744364.90
    Compute (SM) Throughput           %        58.47
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.


## LaunchStats (occupancy)

[ssm-user@ip-10-99-0-199 ~]$ /opt/nvidia/nsight-compute/2025.4.1/ncu   --import /home/ssm-user/roofline_decode_gemv.ncu-rep   --section LaunchStats
[4120] python3.12@127.0.0.1
    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    64
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  12800
    Registers Per Thread             register/thread              96
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block             272
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                  1024
    Threads                                   thread          819200
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                  32
    -------------------------------- --------------- ---------------
	
## WarpStateStats (occupancy)	

[ssm-user@ip-10-99-0-199 ~]$ /opt/nvidia/nsight-compute/2025.4.1/ncu   --import /home/ssm-user/roofline_decode_gemv.ncu-rep   --section WarpStateStats
[4120] python3.12@127.0.0.1
  enable_if<!T7, void>::type kernel<int, int, __half, __half, __half, float, 0, 1, 1, 0, 6, 0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const __half>, cublasGemvTensorStridedBatched<const __half>, cublasGemvTensorStridedBatched<__half>, float>>(T13) (12800, 1, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        14.39
    Warp Cycles Per Executed Instruction           cycle        14.49
    Avg. Active Threads Per Warp                                31.22
    Avg. Not Predicated Off Threads Per Warp                    30.48
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 3.915%
          On average, each warp of this workload spends 6.8 cycles being stalled waiting for the L1 instruction queue
          for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing
          local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to
          avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, or if
          the kernel has excessive register pressure causing spills. If applicable, consider combining multiple
          lower-width memory operations into fewer wider memory operations and try interleaving memory operations and
          math instructions. This stall type represents about 47.3% of the total average of 14.4 cycles between
          issuing two instructions.
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on
          sampling data. The Profiling Guide
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details
          on each stall reason.


# ncu (gemm)

[ssm-user@ip-10-99-0-199 ~]$ /opt/nvidia/nsight-compute/2025.4.1/ncu   --import /home/ssm-user/roofline_decode_gemm.ncu-rep   --section SpeedOfLight
[3675] python3.12@127.0.0.1
  turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn (10, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.99
    SM Frequency                    Mhz       585.01
    Elapsed Cycles                cycle       393063
    Memory Throughput                 %        38.47
    DRAM Throughput                   %        24.57
    Duration                         us       671.87
    L1/TEX Cache Throughput           %        76.95
    L2 Cache Throughput               %        27.71
    SM Active Cycles              cycle    388792.40
    Compute (SM) Throughput           %        83.64
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.







-------------------------------------------------------------------------------
PREFILL
-------------------------------------------------------------------------------


# Prefill analysis

This is a good prefill trace: you’re seeing exactly what you should see for a transformer forward pass on a T4 (sm75): a small set of heavyweight GEMMs + an attention kernel, plus a bunch of light elementwise ops.

1) What the GPU kernel mix is telling you (this is the important part)

From cuda_gpu_kern_sum:

The big rocks

GEMMs dominate:

- turing_fp16_s1688gemm_fp16_256x128... → 39.2%
- turing_fp16_s1688gemm_fp16_128x256... → 21.9%
- Combined: ~61% of GPU time is Tensor Core FP16 matmul kernels.

Attention is real but not #1:

- fmha_cutlassF_f16_aligned_32x128_rf_sm75(...) → 9.3%
- That’s a fused multi-head attention (memory-efficient attention style) kernel.

The small rocks

- Everything else is PyTorch housekeeping:
- layernorm / tanh / add / pow / cat copies / small elementwise kernels
- lots of tiny kernels add overhead and can matter in decode (less so in prefill when big GEMMs dominate).

Interpretation: Your “prefill” path is behaving compute-heavy (as expected) and is strongly GEMM-driven.


# Output


[ssm-user@ip-10-99-0-199 ~]$ nsys profile   --trace=cuda,nvtx,osrt   --stats=true   --capture-range=cudaProfilerApi   --capture-range-end=stop   --force-overwrite true   -o day2_decode   /opt/pytorch/bin/python3 day2-afternoon.py prefill --tool nsight
/opt/pytorch/lib64/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████| 453/453 [00:01<00:00, 324.86it/s, Materializing param=model.layers.31.self_attn.v_proj.weight]
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-1b1c.qdstrm'
[1/8] [0%                          ] day2_decode.nsys-repProcessing events...
[1/8] [========================100%] day2_decode.nsys-rep
Processing 4119 events:
[2/8] [========================100%] day2_decode.sqlite
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style                  Range
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  -----------------------------------
     50.0        270298340          1  270298340.0  270298340.0  270298340  270298340          0.0  PushPop  :DAY2_STEP4_NSIGHT_RUN
     50.0        270184094          1  270184094.0  270184094.0  270184094  270184094          0.0  PushPop  :PREFILL-DOMINANT
      0.0            69951          2      34975.5      34975.5      29051      40900       8378.5  PushPop  CCCL:cub::DeviceScan::InclusiveScan

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)  Name
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----
    100.0        431741084         25  17269643.4  10063912.0  10059793  100124403   24935865.4  poll

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)              Name
 --------  ---------------  ---------  ----------  --------  --------  ---------  -----------  ----------------------------
     96.5        228706666          8  28588333.3    5848.0      3731  228659671   80841028.4  cudaStreamSynchronize
      3.3          7856659       1073      7322.1    6892.0      5331      32535       2241.9  cudaLaunchKernel
      0.1           222687       1073       207.5     196.0       137        811         68.6  cuKernelGetName
      0.1           205659         17     12097.6    9709.0      5034      34268       7410.2  cudaMemcpyAsync
      0.0            37011         33      1121.5    1089.0      1049       1672        108.7  cudaStreamIsCapturing_v10000
      0.0            20890          1     20890.0   20890.0     20890      20890          0.0  cudaDeviceSynchronize
      0.0             8810          1      8810.0    8810.0      8810       8810          0.0  cuProfilerStart

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     39.2        104053096        160   650331.8   445868.0    290557   1866000     475172.1  turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn
     21.9         58211773         32  1819117.9  1882464.0   1370420   2120525     222079.3  turing_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_stages_32x1_tn
      9.3         24590665         32   768458.3   816377.0    489563    934008     132852.5  fmha_cutlassF_f16_aligned_32x128_rf_sm75(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, c…
      6.3         16725967         96   174228.8   173439.0    170846    180222       2341.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<c10::Half, c10::Ha…
      5.2         13792450        192    71835.7    83679.5     34592     98879      22268.0  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>,…
      4.6         12259191         96   127699.9    65583.5     62271    261150      89948.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::Half>, std:…
      3.1          8157624         32   254925.8   253725.5    251838    263038       3128.1  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::Half, c10::Ha…
      2.1          5582989         32   174468.4   173615.0    172190    179934       2296.1  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      2.1          5580909         32   174403.4   173758.5    171039    180062       2274.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::tanh_kernel_cuda(at::TensorItera…
      2.1          5578834         32   174338.6   173422.0    171710    180542       2415.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<c10::Half>…
      1.5          4057533        128    31699.5    31616.0     21248     37344       3664.2  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.8          2219183         64    34674.7    35696.0     29024     37984       2681.2  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.7          1886544         33    57168.0    58368.0     48032     63039       4087.9  void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::Half, float, (bool)0>(int, T2, const …
      0.5          1350323         64    21098.8    21263.5     19104     22304        782.1  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.4          1062039          1  1062039.0  1062039.0   1062039   1062039          0.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, (b…
      0.0            26303          1    26303.0    26303.0     26303     26303          0.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<…
      0.0            22208          1    22208.0    22208.0     22208     22208          0.0  void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon…
      0.0             9120          2     4560.0     4560.0      3744      5376       1154.0  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      0.0             7232          4     1808.0     1904.0      1376      2048        296.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<bool>, std::array<ch…
      0.0             6784          2     3392.0     3392.0      3360      3424         45.3  void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(…
      0.0             6368          1     6368.0     6368.0      6368      6368          0.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrappe…
      0.0             5536          2     2768.0     2768.0      2560      2976        294.2  void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_…
      0.0             4864          1     4864.0     4864.0      4864      4864          0.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrappe…
      0.0             4832          2     2416.0     2416.0      2208      2624        294.2  void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<long>, std::array<ch…
      0.0             4832          2     2416.0     2416.0      2336      2496        113.1  void at::native::vectorized_elementwise_kernel<(int)2, at::native::BinaryFunctor<long, long, long, …
      0.0             4576          2     2288.0     2288.0      2240      2336         67.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<bool, bool, bool, …
      0.0             4416          1     4416.0     4416.0      4416      4416          0.0  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      0.0             4256          2     2128.0     2128.0      1632      2624        701.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<long, long, bool, …
      0.0             4256          3     1418.7     1184.0      1152      1920        434.5  void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<ch…
      0.0             3968          1     3968.0     3968.0      3968      3968          0.0  void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<long, long, long, at::native…
      0.0             3936          2     1968.0     1968.0      1536      2400        610.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<long, long, bool, …
      0.0             3360          1     3360.0     3360.0      3360      3360          0.0  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>,…
      0.0             3040          2     1520.0     1520.0      1504      1536         22.6  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctorOnSelf_add<long>, std…
      0.0             2848          2     1424.0     1424.0      1376      1472         67.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::float16_copy_kernel_cuda(at::Ten…
      0.0             2784          2     1392.0     1392.0      1376      1408         22.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, floa…
      0.0             2560          1     2560.0     2560.0      2560      2560          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::cos_kernel_cuda(at::TensorIterat…
      0.0             2400          1     2400.0     2400.0      2400      2400          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::TensorIterat…
      0.0             2336          1     2336.0     2336.0      2336      2336          0.0  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctorOnOther_add<long>, st…
      0.0             2336          1     2336.0     2336.0      2336      2336          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::bitwise_not_kernel_cuda(at::Tens…
      0.0             2304          1     2304.0     2304.0      2304      2304          0.0  void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctor_add<long>, std::arra…
      0.0             2080          2     1040.0     1040.0      1024      1056         22.6  void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<lon…
      0.0             1792          1     1792.0     1792.0      1792      1792          0.0  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<long>…
      0.0             1760          1     1760.0     1760.0      1760      1760          0.0  void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTens…
      0.0             1600          1     1600.0     1600.0      1600      1600          0.0  void at::native::unrolled_elementwise_kernel<void at::native::compare_scalar_kernel<long>(at::Tenso…
      0.0             1536          1     1536.0     1536.0      1536      1536          0.0  void at::native::vectorized_elementwise_kernel<(int)2, at::native::<unnamed>::masked_fill_kernel(at…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     62.9            12256      9    1361.8    1184.0      1152      2016        341.3  [CUDA memcpy Device-to-Device]
     29.7             5792      5    1158.4     896.0       864      2112        536.6  [CUDA memcpy Device-to-Host]
      7.4             1440      3     480.0     384.0       352       704        194.6  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      0.025      9     0.003     0.000     0.000     0.008        0.004  [CUDA memcpy Device-to-Device]
      0.000      3     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Host-to-Device]
      0.000      5     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]

Generated:
	/home/ssm-user/day2_decode.nsys-rep
	/home/ssm-user/day2_decode.sqlite
[info] prompt_tokens=1024 (capped at 1024)
[hint] For cleaner Nsight stats (exclude model load/tokenize), run:  nsys profile --trace=cuda,nvtx,osrt --stats=true --capture-range=nvtx --capture-range-end=stop --force-overwrite true -o day2 /opt/pytorch/bin/python3 day2-afternoon.py prefill --tool nsight
[ssm-user@ip-10-99-0-199 ~]$


# ncu output (gemm)



```
/opt/nvidia/nsight-compute/2025.4.1/ncu   --target-processes all   --set roofline   --kernel-name-base demangled   --kernel-name regex:turing_fp16_s1688gemm   --launch-skip 0   --launch-count 1   -o roofline_gemm   /opt/pytorch/bin/python3 day2-afternoon.py  --tool nsight
```

```
[ssm-user@ip-10-99-0-199 ~]$ /opt/nvidia/nsight-compute/2025.4.1/ncu \
  --import /home/ssm-user/roofline_gemm.ncu-rep \
  --section SpeedOfLight
[3036] python3.12@127.0.0.1
  turing_fp16_s1688gemm_fp16_256x128_ldg8_relu_f2f_tn (10, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.98
    SM Frequency                    Mhz       585.01
    Elapsed Cycles                cycle       393042
    Memory Throughput                 %        38.58
    DRAM Throughput                   %        24.59
    Duration                         us       671.84
    L1/TEX Cache Throughput           %        77.16
    L2 Cache Throughput               %        27.70
    SM Active Cycles              cycle    388177.62
    Compute (SM) Throughput           %        83.87
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.
````		  

# ncu output (attention)

```
/opt/nvidia/nsight-compute/2025.4.1/ncu \
  --target-processes all \
  --set roofline \
  --kernel-name-base demangled \
  --kernel-name regex:fmha_cutlass \
  --launch-count 1 \
  -o roofline_fmha \
  /opt/pytorch/bin/python3 day2-afternoon.py prefill --tool nsight

/opt/nvidia/nsight-compute/2025.4.1/ncu \
  --import /home/ssm-user/roofline_fmha.ncu-rep \
  --section SpeedOfLight
```

```
==PROF== Disconnected from process 3338
==PROF== Report: /home/ssm-user/roofline_fmha.ncu-rep
[3338] python3.12@127.0.0.1
  fmha_cutlassF_f16_aligned_32x128_rf_sm75(AttentionKernel<half_t, Sm75, 1, 32, 128, 128, 1, 1>::Params) (32, 32, 1)x(32, 4, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.96
    SM Frequency                    Mhz       584.97
    Elapsed Cycles                cycle       747450
    Memory Throughput                 %        41.71
    DRAM Throughput                   %         6.50
    Duration                         ms         1.28
    L1/TEX Cache Throughput           %        83.40
    L2 Cache Throughput               %        16.58
    SM Active Cycles              cycle    709839.60
    Compute (SM) Throughput           %        27.82
    ----------------------- ----------- ------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential
          reasons.

```