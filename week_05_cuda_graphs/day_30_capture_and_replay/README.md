# Day 30: Capture and Replay

**Time:** 60 minutes. **Goal:** Separate graph setup cost from steady-state replay.

## Knowledge

A CUDA Graph records operations and dependencies for repeated submission. Instead
of submitting every operation separately from Python, the host launches an
instantiated graph. This can reduce repeated submission overhead, especially for
many short kernels. It does not automatically fuse operations, improve arithmetic
intensity, or change a kernel's memory access pattern.

Warm up on a side stream before capture, then establish stream dependencies so
capture sees completed initialization. Capture only GPU-compatible operations:
host synchronization, extracting a device scalar with `.item()`, and data-dependent
Python decisions are not ordinary replayable work. The graph and captured tensor
storage must remain alive for replay.

Separate warmup, capture/instantiation, and replay measurements. A fast replay
can still be a poor choice for a workload that rarely repeats. Compare steady-state
eager and replay wall time, and separately compare device event time. Do not divide
wall time by device time and call the result a speedup. The solution checks multiple
changed inputs before timing replay; numerical equivalence is a prerequisite, not
an optional consequence of a favorable benchmark.

## References

- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): creation, instantiation, and launch.
- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs): warmup and capture constraints.

## Practice

Complete [exercise.py](exercise.py), then inspect [solution.py](solution.py) and
[shared capture code](../common.py). Record setup and both steady-state timing series.
Estimate how many replays would amortize setup, if the measured saving is positive.

## Run

```bash
python week_05_cuda_graphs/day_30_capture_and_replay/exercise.py
python week_05_cuda_graphs/day_30_capture_and_replay/solution.py
```

**Done when:** Correctness passes and `results/capture.json` separates setup from replay.
