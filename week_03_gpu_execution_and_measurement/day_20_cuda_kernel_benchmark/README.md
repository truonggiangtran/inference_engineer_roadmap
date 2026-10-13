# Day 20: Reuse a CUDA Kernel as a Benchmarking Exercise

**Time:** 90 minutes

## Goal

Apply reliable measurement to code you already understand before interpreting whole-model results.

## Knowledge

**SAXPY** (`y = a·x + y`) is the classic memory-bound kernel. Each element does 2 FLOPs and moves 12 bytes: it reads `x`, reads `y`, and writes `y`, using FP32. That is about 0.17 FLOP/byte, far left of any roofline ridge. Its speed therefore measures **effective DRAM bandwidth**:

$$
\text{effective GB/s} = \frac{3 \cdot n \cdot 4\ \text{bytes}}{t\ (\text{s}) \cdot 10^9}
$$

**Measurement discipline, the same as Day 16 but in CUDA C++:**

- **Validate first.** Compare against a CPU reference on the same inputs with a stated tolerance. A fast wrong kernel is worthless.
- **Keep setup outside the timing.** `cudaMalloc`, host-to-device copies, and the correctness run all happen before the timed loop.
- **Warm up.** The first launch pays for module loading and clock ramp-up.
- **Use CUDA events.** `cudaEventRecord(start)`, the kernel, `cudaEventRecord(stop)`, `cudaEventSynchronize(stop)`, then `cudaEventElapsedTime`.
- **Repeat and report the median and spread** for at least two sizes. A small input (`2^16`) is dominated by launch overhead and won't reach bandwidth. A large input (`2^26`) approaches the DRAM peak.
- **Check every CUDA call.** The `CUDA_CHECK` macro and `cudaGetLastError()` after each launch catch silent failures.

**Compile for the RTX 5070:** `-arch=sm_120` generates native code for compute capability 12.0. This requires CUDA toolkit 12.8 or later.

**Two NVIDIA profilers with different purposes:**

| Tool | Answers | Output |
| --- | --- | --- |
| **Nsight Systems** (`nsys`) | *When* things happen: the timeline of CPU calls, launches, copies, and kernels, and the gaps between them | `.nsys-rep` |
| **Nsight Compute** (`ncu`) | *How well* one kernel uses the hardware: achieved bandwidth, occupancy, stalls, and roofline | `.ncu-rep` |

`ncu` needs access to GPU performance counters. If it reports `ERR_NVGPUCTRPERM`, or doesn't yet support your GPU, record that. **Don't claim** that occupancy or bandwidth counters were measured when they weren't.

**Why this doesn't prove an LLM speedup.** A microbenchmark isolates one kernel with ideal, contiguous, warmed-up data. A model step involves hundreds of different kernels, launch gaps, and Python overhead. Improving one kernel helps only in proportion to its share of the step time (Amdahl's law), so measure the whole model separately.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): performance metrics, timing with events, and effective bandwidth.
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): the `nsys profile` command and timeline view.
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): sections, metrics, and roofline.
- [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html): Blackwell (`sm_120`) compiler support.

## Practice

Complete [exercise.cu](exercise.cu):

1. Implement the `saxpy` kernel body.
2. Implement the warmup and CUDA-event timing loop.
3. Run both sizes. The program validates against a CPU reference and writes `results/saxpy.csv` with the median, 10th percentile, 90th percentile, and effective GB/s.
4. Optionally capture a timeline and kernel metrics:

```bash
cd week_03_gpu_execution_and_measurement/day_20_cuda_kernel_benchmark
mkdir -p results
nvcc -O3 -arch=sm_120 -o saxpy_exercise exercise.cu && ./saxpy_exercise
# reference: nvcc -O3 -arch=sm_120 -o saxpy solution.cu && ./saxpy

# timeline (Nsight Systems)
nsys profile -o results/saxpy --force-overwrite true ./saxpy_exercise
# kernel metrics (Nsight Compute); may need counter permissions
ncu --set full -k saxpy -o results/saxpy --force-overwrite ./saxpy_exercise
```

Profiler outputs and binaries are excluded by git; the CSV is committed.

## Done when

You have a trustworthy small benchmark report and can explain why its microbenchmark result does not directly establish an LLM speedup.
