# Day 29: Eager Launch Baseline

**Time:** 60 minutes. **Goal:** Establish timing boundaries before adding graphs.

## Knowledge

CUDA work is asynchronous with respect to Python. A host timer around a tensor
operation can measure submission rather than completion. Synchronizing before the
timer excludes previously queued work; synchronizing afterward waits for the
submitted workload. This synchronized wall measurement includes Python, submission,
and completion-wait overhead. CUDA events measure elapsed device timeline time
between event markers instead. Neither boundary should be substituted for the other.

The shared workload transforms an FP32 vector `[N]` through eight multiply/add
steps, using persistent temporary and output buffers. Preallocation prevents buffer
creation from becoming an accidental difference between eager and replay modes.
Warmup is outside the timed samples. Record the GPU, software versions, vector
length, operation count, and all samples, not just the fastest observation.

This workload deliberately contains short operations. It is a tool for studying
launch overhead, not a representative LLM or an optimized fused kernel. A longer
GPU duration at larger `N` may change which overhead dominates. Establish the
baseline before predicting any graph speedup.

## References

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous execution and synchronization.
- [CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html): event timing.

## Practice

1. Complete [exercise.py](exercise.py); compare with [solution.py](solution.py).
2. Collect 30 warmed-up samples, recording wall and device time separately.
3. Explain which allocations and transfers are excluded. Keep measurements blank without CUDA.

## Run (Linux, from workspace root)

```bash
python week_05_cuda_graphs/day_29_eager_baseline/exercise.py
python week_05_cuda_graphs/day_29_eager_baseline/solution.py
```

**Done when:** `results/eager.json` exists from a GPU run and you can explain both timing boundaries.
