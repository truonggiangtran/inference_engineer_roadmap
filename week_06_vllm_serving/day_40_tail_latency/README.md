# Day 40: Tail Latency and Load

**Time:** 60 minutes. **Goal:** Interpret distributions without overstating tail evidence.

## Knowledge

A median describes a typical sample but hides slow requests. A p95 sample
quantile summarizes the upper part of a distribution, yet its reliability depends
on sample count, independence, workload, and measurement conditions. In a tiny
run, only a few observations determine the apparent tail. A reported decimal
place is not evidence of statistical confidence.

Inspect raw samples and run order as well as summary values. Startup contamination,
cache warming, thermal changes, long prompts, and early EOS can create different
subpopulations. A single pooled percentile may conceal these effects. Repeat
conditions and retain every successful observation rather than selecting a
favorable run. Failed requests should be reported explicitly, not discarded.

This week's load generator is closed loop. Workers wait for responses before
processing more work, so it cannot show the full queue buildup caused by an
independent arrival stream. Do not turn its p95 into a production SLO guarantee.
If overload behavior is the next question, use an explicitly rate-controlled
arrival process and include scheduled-arrival queueing in the measurement boundary.

## References

- [Python statistics](https://docs.python.org/3/library/statistics.html): descriptive summaries.
- [Client measurement contract](../common/README.md): load and latency boundaries.

## Practice and run

Implement [exercise.py](exercise.py)'s interpolation. Run at least three larger
closed-loop trials (for example 256 requests), then analyze each file with
[solution.py](solution.py). Keep trial identity rather than hiding variation by pooling.
Use `--output` with a unique analysis filename for each trial.

```bash
python week_06_vllm_serving/day_40_tail_latency/exercise.py
python week_06_vllm_serving/day_40_tail_latency/solution.py \
  week_06_vllm_serving/day_38_concurrency_sweep/results/vllm-c1.json
```

**Done when:** Your report distinguishes descriptive sample tails from a validated service guarantee.
