# Day 15: Map Model Operations to GPU Work

**Time:** 60 minutes

## Goal

Understand the boundary between Python orchestration, tensor-library calls, and GPU kernels.

## Knowledge

**Three layers of execution:**

```text
Python (model.forward, loops)           CPU: decides what to run
  └─ PyTorch operators (aten::linear, aten::mul, ...)    CPU: dispatch, shape checks, allocation
       └─ GPU kernels (cuBLAS GEMM, elementwise, fused attention ...)   GPU: actual math
```

A single Python line can become zero kernels (a `view`), one kernel (a `mul`), or several (an unfused RMSNorm: `pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`). A fused kernel, such as SDPA's flash or memory-efficient backend, can replace several. **Count real kernels in a trace; don't assume one per Python operation.**

**Mapping model operations to operation types:**

| Model operation | Typical aten op(s) | Kind of GPU work |
| --- | --- | --- |
| Q/K/V/O projections, MLP, `lm_head` | `aten::linear` → `aten::mm`/`addmm` | GEMM (cuBLAS) |
| Attention | `aten::scaled_dot_product_attention` → flash/efficient/math | Fused attention or batched GEMMs plus softmax |
| RMSNorm | `pow`, `mean`, `rsqrt`, `mul` (or a fused `rms_norm`) | Reduction plus elementwise |
| Residual add, SiLU, gating, RoPE | `add`, `mul`, `silu`, `neg`, `cat` | Elementwise and copies |
| Embedding | `aten::embedding` → `index_select` | Gather (indexing) |

**Asynchronous launches.** The CPU queues kernels and moves on. As long as the CPU stays ahead, the GPU is fed. When per-kernel CPU overhead exceeds the kernel's GPU time, which is common for small decode-step kernels, the GPU sits idle between launches. A trace makes this visible (Day 17).

**Define what you time.** These are different measurements:

| Metric | Includes |
| --- | --- |
| CPU time of an operator | Dispatch and launch overhead on the host |
| GPU timeline time | Interval between GPU events on a stream, including idle gaps |
| GPU prefill latency | Forward pass only, from the token tensor already on the GPU to the logits |
| End-to-end time to first token | Tokenization + transfer + prefill + token selection + detokenization (+ queuing in a server) |

## References

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous execution and streams.
- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): the performance-measurement and host-device sections.
- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`.

## Practice

Complete [exercise.py](exercise.py):

1. `classify_op()`: map an aten operator name to `matmul`, `attention`, `normalization`, `elementwise`, `copy`, `indexing`, or `other`.
2. `profile_forward()`: profile one warmed-up prefill forward with CPU and CUDA activities.
3. `summarize()`: total the CPU time per category, and count aten operator calls compared with GPU kernel launches.
4. `end_to_end_breakdown()`: time tokenization, the host-to-device copy, the GPU forward (CUDA events), and token selection with decoding, so you can see how the prefill forward differs from end-to-end time to first token.

```bash
python week_03_gpu_execution_and_measurement/day_15_operations_to_gpu_work/exercise.py
```

On CPU the script still classifies operators, but there are no GPU kernels to count.

## Done when

You can explain how high-level operations reach the GPU and distinguish CPU time, GPU timeline time, and end-to-end request latency.
