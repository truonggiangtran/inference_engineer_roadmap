# Day 26: Create Plots and One Optional Experiment

**Time:** 60 minutes

## Goal

Turn raw measurements into an understandable comparison without expanding the required project.

## Knowledge

**A good benchmark plot answers one question** and carries enough context to be read on its own:

- **Axes and units:** for example, "ms per decode step" and "MiB".
- **Workload:** prompt lengths, 32 teacher-forced steps, and batch size 1.
- **Precision and device:** for example, BF16 on an RTX 5070.
- **Sample count and variability:** `n = 10` measured requests, with error bars showing the 10th to 90th percentiles around the median. Say which in the caption, because error bars are ambiguous otherwise.
- **Comparison groups:** use the same colors for cached and uncached in every plot.

Grouped bar charts are appropriate when there are only two prompt lengths. Avoid line plots that suggest values *between* 128 and 512 that you didn't measure.

**Results table.** It follows the Results Table Template in the roadmap. Keep the following clearly separated:

- **Measured** values: prefill median, decode median per step, and peak allocated memory.
- **Derived** values: tokens/s, which equals 32 ÷ decode seconds.
- **Estimated** values: the analytical KV size.

Don't extrapolate to other models, GPUs, batch sizes, or serving throughput.

**Optional stretch experiment.** Attempt it only if the core comparison is complete and correct. Choose one:

- Batch size 1 compared with 2.
- BF16 compared with FP32.

Change exactly one factor, repeat the Day 23 correctness checks for the new setting, and document the extra tradeoff:

- **Batch size 2:** throughput compared with per-request latency, and the cache memory doubling.
- **FP32:** speed and memory compared with numerical error.

If you skip it, spend the time improving the core results and plots instead.

## References

- [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html): figures, axes, `bar`, and `errorbar`.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): reduced-precision notes, for the optional dtype experiment.
- [Cache strategies](https://huggingface.co/docs/transformers/kv_cache): needed only for an optional cache variant.

## Practice

Complete [exercise.py](exercise.py). It reads the Day 23, Day 24, and Day 25 `results/` files.

1. `grouped_bars()`: draw a labeled grouped bar chart with optional asymmetric error bars, and save it.
2. The script plots the decode time per step, the prefill latency (both from timings), and the peak allocated memory (from the memory results).
3. `results_table()`: build the Markdown results table with a correctness column taken from Day 23.

Outputs (committed):

- `results/decode_ms_per_step.png`
- `results/prefill_ms.png`
- `results/peak_memory.png`
- `results/results_table.md`

```bash
python week_04_inference_study/day_26_plots_and_optional_experiment/exercise.py
```

## Done when

The primary study has readable plots and a results table, whether or not a stretch experiment was attempted.
