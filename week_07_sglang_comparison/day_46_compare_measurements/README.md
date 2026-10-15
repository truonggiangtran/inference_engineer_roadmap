# Day 46: Compare Measurements

**Time:** 60 minutes. **Goal:** Reject incompatible results before comparing performance.

## Knowledge

A comparison requires compatible inputs and boundaries, not just two latency
numbers. The shared client records a manifest hash, model identity, generation
settings, concurrency, warmup count, and actual token accounting. The reference
gate compares these fields and the common metadata before generating a table.
Different labels identify the two runtime configurations.

The gate is intentionally conservative about output lengths. If EOS or numerical
differences produce different generation lengths, the run is not matched work.
Investigate or present it separately rather than quietly interpreting throughput
as a controlled runtime speedup. Equal token lengths still do not prove identical
kernel math or matching cache states; the experiment worksheets carry those
additional assumptions.

Compare throughput and latency together, preserving sample counts and repeat
identity. Higher tokens per second may accompany slower first output or worse
tail behavior. A universal ranking cannot be derived from one small model,
device, prompt family, and concurrency. Report default-versus-default comparisons
as such, and distinguish them from feature-isolated experiments. Keep the raw
records so a reviewer can check the table rather than relying on a plotted ratio.

## References

- [Shared schema and boundaries](../../week_06_vllm_serving/common/README.md).
- Your Day 42 report and Day 45 cache protocol.

## Practice and run

Implement [exercise.py](exercise.py)'s compatibility gate using actual result
files; deliberately alter one token count to verify rejection. Compare with
[solution.py](solution.py):

```bash
python week_07_sglang_comparison/day_46_compare_measurements/solution.py \
  week_06_vllm_serving/day_38_concurrency_sweep/results/vllm-c1.json \
  week_07_sglang_comparison/day_44_matched_workload/results/sglang-c1.json
```

Use `--output` with a unique comparison filename for each concurrency and trial.

**Done when:** `results/comparison.json` comes from compatible runs, or the mismatch is explicitly investigated and reported.
