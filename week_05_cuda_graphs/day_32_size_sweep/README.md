# Day 32: Workload Size Sweep

**Time:** 60 minutes. **Goal:** Test when launch overhead matters.

## Knowledge

Keep the number of operations constant while varying the vector length. Small
operations may finish quickly enough that repeated host submission leaves gaps
in the GPU timeline. At larger sizes, memory traffic and execution can dominate.
A graph may reduce those gaps without changing the duration of an individual
kernel. A fused implementation would be a different experiment and should not
be introduced into only one side of this comparison.

For matching wall-time boundaries, define speedup as eager median divided by
replay median. Values below one are valid negative results. Record raw samples
and spread; a tiny difference within run-to-run noise is not decisive evidence.
Capture and warmup costs remain separate from both steady-state distributions.

The reference runs eager before replay. Repeat the experiment in reversed order
before making a strong claim, because clocks, thermal state, and background work
can drift. Do not compare values across devices as if they were one controlled
sweep. The expected outcome is an evidence-backed explanation, not a required
speedup or a preselected threshold.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): measurement and bottlenecks.
- [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): repeatability.

## Practice

Complete [exercise.py](exercise.py)'s ratio check. Run [solution.py](solution.py)
at four sizes, keep launch count fixed, then repeat in reverse execution order.
Create a small table of wall/event medians and setup cost. Explain one inconclusive result.

## Run

```bash
python week_05_cuda_graphs/day_32_size_sweep/exercise.py
python week_05_cuda_graphs/day_32_size_sweep/solution.py
```

**Done when:** `results/sweep.json` supports a qualified overhead hypothesis.
