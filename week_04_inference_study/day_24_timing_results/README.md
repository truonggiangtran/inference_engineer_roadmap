# Day 24: Collect Repeatable Timing Results

**Time:** 60 minutes

## Goal

Measure the cache comparison under identical, explicitly defined workloads.

## Knowledge

**Four primary conditions:** {128, 512 prompt tokens} × {uncached, cached}, at batch size 1, with the same 32-token teacher-forced continuation.

**Metrics and definitions, which must match your Day 21 protocol:**

| Metric | Definition |
| --- | --- |
| Prefill latency (ms) | CUDA-event interval from just before the prompt forward to just after it |
| Total decode time (ms) | Event interval covering all 32 one-token steps |
| Average time per decode step (ms) | Total decode time ÷ 32 |
| Decode tokens/s | 32 ÷ (total decode time in seconds) |

**What the intervals include.** `PhaseTimer` records events on the GPU stream, so each interval covers:

- All GPU kernels.
- Any **GPU idle time** while Python and the dispatcher launch the next kernel. Python orchestration *is* included whenever it starves the GPU.

The intervals do **not** include:

- Tokenization (it happened on Day 22).
- Host-to-device copies (the inputs are already on the GPU).
- Model loading.

These numbers measure *model-forward throughput for one sequence*. They are not a serving system's request throughput, which also involves batching, scheduling, queuing, and networking.

**Repeat-measurement discipline:**

- Run at least 3 warmup requests per condition, then at least 10 measured requests.
- Give each request a fresh cache.
- Run nothing else on the GPU, and keep the same power and clock settings.
- Report the **median** and spread (10th to 90th percentiles). Never report only the fastest run.
- Ten samples can't support tail-latency claims such as p99.
- If the first request in a process is slow, record that cold start separately; don't mix it into the warmed-up statistics.

**Expectation to test, not assume.** With a cache, each decode step should cost roughly the same regardless of how many tokens came before. Uncached steps should get slower as context grows, because each one reprocesses the full prefix. For a 0.6B model at batch 1, however, fixed per-step overhead (launches and Python) may dominate both modes and shrink the visible gap. Report what you observe.

## References

- [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): warmup and statistics.
- [`torch.cuda.Event`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html): what the events measure.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): synchronization.

## Practice

Complete [exercise.py](exercise.py):

1. `time_condition()`: run the warmup requests, then collect `(prefill_ms, decode_ms)` from each measured request using `PhaseTimer`. Record the first warmup request's total time as `first_request_ms`. Only the first condition run in the process includes true cold-start costs.
2. `summarize()`: compute the median, 10th and 90th percentiles, and number of samples. Derive the time per step and tokens per second from the median decode time.
3. The script saves `results/timings.json` with the raw samples, summaries, environment, and the definitions of the timing boundaries.

```bash
python week_04_inference_study/day_24_timing_results/exercise.py
```

On CPU, the script runs with fewer repeats and labels the results as CPU-only. Those numbers are not GPU results.

## Done when

All four primary conditions have repeat measurements and defined metrics, including exactly which CPU overheads and GPU work the timings include.
