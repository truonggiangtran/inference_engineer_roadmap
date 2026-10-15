# Day 41: Graph Execution Ablation

**Time:** 90 minutes. **Goal:** Test graph-enabled serving without overclaiming kernel effects.

## Knowledge

Serving runtimes often combine graphs with compilation, fused operators, specialized
attention kernels, and scheduler policies. A graph-related switch may alter more
than one execution path. Read the installed release's option semantics and startup
logs before describing the experiment as a pure CUDA Graph ablation.

For vLLM, an eager-enforcement option is a useful contrasting configuration, but it
can affect compilation and graph execution together. If you cannot isolate graph
replay while preserving everything else, name the comparison "default versus
enforced eager" and disclose the coupling. Do not attribute all differences to
graph launch overhead.

Compare the same manifest, actual token lengths, concurrency, dtype, model revision,
and cache protocol. Graph capture and initialization can add startup cost and
memory consumption, which belong outside steady-state latency but inside the
operational report. Repeat conditions with reversed order. A short-model serving
speedup does not imply that each kernel became faster; use separate diagnostic
evidence to explain whether submission gaps changed. Keep profiler runs outside
the reported benchmark measurements.

## References

- [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/): `--enforce-eager` and compilation settings.
- [Week 5 graph findings](../../week_05_cuda_graphs/day_35_graph_review/worksheet.md).

## Practice and run

Use [worksheet.md](worksheet.md). Run the Day 36 server command first normally,
then with `--enforce-eager`, verifying the installed help and actual backend logs.
Stop each foreground server before the other starts. Repeat identical
[client runs](../common/README.md) with unique labels and filenames.

**Done when:** Your evidence separates setup, steady state, and coupled execution-setting changes.
