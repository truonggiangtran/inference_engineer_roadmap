# Day 18: Compute, Bandwidth, and Launch Overhead

**Time:** 60 minutes

## Goal

Build evidence-based performance hypotheses rather than labeling every inference workload the same way.

## Knowledge

**Arithmetic intensity (AI)** is the number of FLOPs performed per byte moved to or from DRAM. For a linear layer `[M, K] × [K, N]`:

$$
\text{FLOPs} = 2MKN \qquad \text{min bytes} \approx (MK + KN + MN)\cdot b \qquad \text{AI} = \frac{\text{FLOPs}}{\text{bytes}}
$$

Here `b` is bytes per element (2 for BF16). The weight term `KN` usually dominates. When `M = 1`, as in a decode step at batch 1, each weight byte is used for only about one multiply-add, so AI is about 1 FLOP/byte. When `M = 512`, as in prefill, each weight is reused 512 times.

**The roofline.** A kernel is limited by whichever is lower:

$$
\text{attainable FLOP/s} = \min(\text{peak FLOP/s},\ \text{AI} \times \text{peak bandwidth})
$$

The *ridge point* `peak FLOP/s ÷ peak bandwidth` separates **bandwidth-bound** (left) from **compute-bound** (right). Look up your RTX 5070's peak BF16 tensor throughput and memory bandwidth in NVIDIA's specifications, record them with their source in the script constants, and note that the achievable peaks are lower than the advertised ones.

**Three hypotheses to choose between:**

| Hypothesis | Typical evidence |
| --- | --- |
| **Compute-bound** | Achieved TFLOP/s near peak; time scales with FLOPs |
| **Bandwidth-bound** | Achieved GB/s near peak; time scales with bytes (weights + cache) |
| **Launch / CPU-overhead-bound** | Kernels far below both peaks; time ≈ number of kernels × microseconds; GPU timeline has gaps |

For a **small** model like 0.6B, decode can be dominated by launch and Python overhead even though the textbook answer for decode is "bandwidth-bound". Many layers of tiny kernels each take only a few microseconds of GPU time. Always state what observation would **disprove** your hypothesis.

**Estimates are assumptions.** The minimum-bytes formula assumes each operand is read once. Several factors move real traffic in either direction:

- Tiling and cache reuse.
- Fusion.
- Split-K GEMMs.
- Implementation choices.

**Precision matters.** FP32, TF32, FP16, and BF16 differ in both numerics and speed. BF16 and FP16 use tensor cores. TF32 is an FP32-range format with reduced mantissa precision, controlled by `torch.backends.cuda.matmul.allow_tf32`. Never compare throughput across dtypes as if they were the same workload.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): the memory-throughput and effective-bandwidth sections.
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): the roofline and memory-workload analysis.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): TF32 and reduced-precision matmul notes.

## Practice

Complete [exercise.py](exercise.py):

1. `linear_flops()`, `linear_min_bytes()`, and `arithmetic_intensity()`: apply the formulas above.
2. `measure_linear()`: time `F.linear` at `M ∈ {1, 16, 128, 512}` with the model's real `q_proj` and MLP `gate_proj` sizes, then compute the achieved TFLOP/s and GB/s.
3. `launch_overhead()`: estimate per-kernel GPU time and per-launch host time using a 1-element `add`.
4. Fill in `PEAK_TFLOPS` and `PEAK_GBPS` with sourced values. The script then reports the ridge point and, for each measured shape, whether it falls below or above it.

Finally, write a hypothesis for prefill and one for decode, using your Day 17 trace. Include the observation that would disprove each one.

```bash
python week_03_gpu_execution_and_measurement/day_18_compute_bandwidth_overhead/exercise.py
```

## Done when

You can explain a plausible bottleneck with supporting evidence and acknowledge that FP32, TF32, FP16, and BF16 do not imply identical numerical or performance behavior.
