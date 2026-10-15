# Day 34: Profile and Explain

**Time:** 90 minutes. **Goal:** Connect timing observations to execution evidence.

## Knowledge

A timeline helps distinguish CPU submission, GPU kernels, gaps, and synchronization.
Label eager and replay regions so their launch patterns can be compared. CUDA Graph
replay changes how work is submitted; it does not imply that all device operations
have become one fused kernel. Depending on profiler support, graph activity may
be displayed at different levels of detail.

Collect diagnostic traces separately from benchmark samples. Profiling can alter
timing and create overhead, especially for short kernels, so a traced duration is
not a replacement for Day 32's uninstrumented benchmark. Warm up and capture before
starting the profiler to avoid confusing setup activity with replay.

Use the trace to test a specific hypothesis, such as fewer host launches or smaller
submission gaps. If graph nodes are not visible with your profiler configuration,
record that limitation rather than inventing an explanation. Nsight Systems can
provide an additional view when available; Nsight Compute answers different,
kernel-level questions. Link each performance claim to both an unprofiled
measurement and suitable diagnostic evidence, and keep large traces local.

## References

- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): activities and regions.
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): graph tracing and timeline interpretation.

## Practice

Implement [exercise.py](exercise.py), compare with [solution.py](solution.py),
and inspect the trace locally. Record three observations and one unsupported claim
you deliberately avoid. Do not commit the large trace.

## Run

```bash
python week_05_cuda_graphs/day_34_profile_and_explain/exercise.py
python week_05_cuda_graphs/day_34_profile_and_explain/solution.py
```

**Done when:** A separate diagnostic trace and short written interpretation explain the measured boundaries.
