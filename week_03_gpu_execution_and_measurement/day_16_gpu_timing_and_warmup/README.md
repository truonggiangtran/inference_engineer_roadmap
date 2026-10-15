# Day 16: Correct GPU Timing and Warmup

**Time:** 60 minutes

## Goal

Avoid misleading timings caused by asynchronous launches and one-time initialization.

## Knowledge

**The asynchronous trap:**

```python
t0 = time.perf_counter()
y = x @ w                 # returns as soon as the kernel is QUEUED
t1 = time.perf_counter()  # measures launch/submission (microseconds), not execution
```

**Three ways to time GPU work:**

| Method | Measures | Pitfall |
| --- | --- | --- |
| Host timer, no sync | Launch or submission time | Looks impossibly fast |
| `synchronize()`, host timer, op, `synchronize()`, host timer | Completed execution plus launch overhead plus sync cost | Correct, but includes host-side overhead |
| CUDA events: `start.record()`, op, `end.record()`, then `end.synchronize()` | GPU stream time between the two events | Includes **idle gaps** on the stream, not only active kernel time |

`start.elapsed_time(end)` returns milliseconds between the points where each event was *reached on the GPU stream*. If the CPU is slow to launch the next kernel, that idle time is counted. Events measure a *timeline interval*, not the sum of kernel durations.

**Warmup.** The first call pays one-time costs:

- CUDA context creation.
- Loading the cuBLAS or cuDNN library and selecting a heuristic.
- The caching allocator obtaining memory from the driver.
- Lazy module initialization.

Run a few warmup iterations, preallocate inputs, and keep allocation and host-to-device transfers **outside** the timed region when measuring an isolated operation. Record the cold-start cost separately if it matters, because it does for serving.

**Statistics.** GPU timings vary because of clocks, boost behavior, other processes, and thermals:

- Report the **median** and a spread measure, such as the 10th to 90th percentiles or the standard deviation.
- Don't report the single best run, which is optimistic and not reproducible.
- `torch.utils.benchmark.Timer` handles warmup, synchronization, and repeated measurement for you. Use it as a cross-check.

## References

- [PyTorch benchmarking recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): `torch.utils.benchmark.Timer` and `blocked_autorange`.
- [`torch.cuda.Event`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html): `enable_timing` and `elapsed_time`.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous execution.

## Practice

Complete [exercise.py](exercise.py) using a BF16 `4096 × 4096` matmul:

1. `time_cold()`: time the very first call with synchronization, before any warmup.
2. `time_unsynchronized()`, `time_synchronized()`, and `time_cuda_events()`: collect repeated samples in milliseconds.
3. `summarize()`: compute the median, the 10th and 90th percentiles, and the standard deviation.
4. Compare all three methods with `torch.utils.benchmark.Timer`, and explain why the unsynchronized median is far too small.

```bash
python week_03_gpu_execution_and_measurement/day_16_gpu_timing_and_warmup/exercise.py
```

On CPU, the script runs a smaller FP32 matmul and labels the results as CPU timings.

## Done when

You can justify the timer and boundaries used for a benchmark and explain which overheads are included.
