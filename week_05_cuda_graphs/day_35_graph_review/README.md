# Day 35: CUDA Graph Review

**Time:** 45 minutes. **Goal:** Defend graph findings without confusing them with kernel optimization.

## Knowledge

Execution optimization changes how work is scheduled or submitted; kernel
optimization changes the work inside a kernel or its memory behavior. CUDA Graphs
primarily address repeated submission overhead. Fusion may reduce both launches
and intermediate memory traffic, while tiling changes data reuse. These are related
but distinct mechanisms that need separate experiments.

A useful graph report records shape, storage lifetime, warmup, capture cost, replay
cost, and numerical checks. It describes whether input updates and synchronization
were inside timing boundaries. A microbenchmark cannot by itself establish a
serving improvement because a server also schedules requests, manages caches,
serializes responses, and handles variable workloads.

Review a negative or inconclusive case as carefully as a positive one. If replay
does not help a large operation, this may be consistent with a workload dominated
by execution rather than submission. If results are noisy, more controlled repeats
may be the correct next step. The purpose of the week is to explain when graphs
are appropriate and what remains unproven before moving to a serving runtime.

## References

- [NVIDIA graph model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
- Your Days 29–34 results and diagnostic notes.

## Practice and run

Fill in [worksheet.md](worksheet.md); no script or model answers substitute for your measurements.

**Done when:** You can explain persistent storage, setup amortization, shape costs, and one limitation in five minutes.
