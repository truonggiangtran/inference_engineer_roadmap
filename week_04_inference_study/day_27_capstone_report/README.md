# Day 27: Assemble the Demonstrable Project

**Time:** 90 minutes

## Goal

Make the work understandable and reproducible for someone reviewing your engineering judgment.

## Knowledge

**A reviewer judges three things:**

- **Reproducibility.** Could they rerun it and get comparable numbers?
- **Honesty.** Are observations clearly separated from hypotheses, and are limitations stated?
- **Reasoning.** Is each claim backed by specific evidence, such as a measurement, a trace, or a calculation?

**Separating evidence levels:**

| Label | Meaning | Example |
| --- | --- | --- |
| **Observed** | Measured directly, with its protocol | "Cached decode median 9.1 ms/step at P=512 (n=10, p10–p90 8.9–9.4)" |
| **Derived** | Computed from observed values | "32 / decode seconds = tokens/s" |
| **Estimated** | Analytical, from stated assumptions | "KV size from the formula with config values" |
| **Hypothesis** | Proposed explanation, with what would disprove it | "Decode is launch-bound; disproved if a CUDA graph doesn't reduce step time" |

**What is specific to this setup:** an RTX 5070, a 0.6B dense model, BF16, the SDPA backend, batch size 1, teacher forcing, and event-based timing that includes Python launch gaps. Results may not transfer to:

- Larger models.
- Other GPUs.
- Batched serving.
- Other attention backends.

**Negative or inconclusive results are acceptable.** If caching helped less than expected at 128 tokens, or the memory peak contradicted the simple story, report it with your best explanation. A speedup isn't required. A trustworthy measurement is.

**Artifacts already produced:**

| Artifact | Location |
| --- | --- |
| Environment | `week_01_.../day_01_.../results/environment.json` |
| Model anatomy | `week_02_.../day_08_.../results/model_anatomy.json` |
| Forward-pass walkthrough | `week_02_.../day_14_.../worksheet.md` |
| Benchmark protocol | `week_03_.../day_21_.../worksheet.md` |
| Workload manifest | `week_04_.../day_22_.../results/workload_manifest.json` |
| Correctness | `week_04_.../day_23_.../results/cache_correctness.json` |
| Timings | `week_04_.../day_24_.../results/timings.json` |
| Memory and diagnostic tables | `week_04_.../day_25_.../results/` |
| Plots and table | `week_04_.../day_26_.../results/` |
| Representative traces (local, git-ignored) | `*.pt.trace.json` from Days 17 and 25 |

## References

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B): model attribution and license.
- [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): describing measurement methods accurately.
- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): describing traces accurately.

## Practice

Fill in [worksheet.md](worksheet.md). It is the capstone report and reproduction guide. You can copy it to the repository root or publish it as the project's README.

There is no model answer for this day, because the report must reflect your own measurements.

## Done when

A reviewer can understand the comparison, rerun it, and distinguish observed results from hypotheses without asking for missing configuration details.
