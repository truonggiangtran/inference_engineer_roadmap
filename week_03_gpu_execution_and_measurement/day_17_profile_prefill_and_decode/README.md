# Day 17: Profile Prefill and Decoding Separately

**Time:** 60 minutes

## Goal

Identify where time goes in the model without confusing profiler output with a clean benchmark.

## Knowledge

**Why separate the phases?** Prefill and decode have different shapes, so they have different bottlenecks:

| | Prefill (`P` tokens) | Cached decode step |
| --- | --- | --- |
| GEMM `M` dimension | `P` | 1 |
| Attention | `[P × P]` scores | `[1 × T_cached]` scores |
| Typical character | Large GEMMs with high arithmetic intensity | Many small kernels. Weights and the cache are read for little math, and launch overhead is significant. |

Profiling them together mixes these two effects. Profile each phase in its own session, or label each phase with `torch.profiler.record_function`.

**Reading a profiler table:**

- **Self time:** time spent in the operator itself, excluding its children.
- **Total (inclusive) time:** includes its children. `aten::linear` contains `aten::addmm` or `aten::mm`.

**Never add inclusive times of nested operators.** You would count the same work twice. Use self time to find where the work actually happens.

- **CPU time** is host-side dispatch.
- **CUDA (device) time** is kernel execution attributed to the operator.

For decode, a large CPU total next to a small CUDA total suggests a launch- or host-bound phase.

**Traces.** `prof.export_chrome_trace(path)` writes a timeline you can open in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. Look for:

- Kernel launches: CPU `cudaLaunchKernel` rows.
- Kernel execution: GPU stream rows.
- Memory copies: `Memcpy` rows.
- **Gaps** on the GPU stream, where it waits for the CPU.

**Profiling adds overhead.** Shape recording, stack collection, and tracing all slow the run. Use profiles to understand *what* happens and *where*. Make speed claims only from separate, unprofiled timing runs (Day 16 methods and Week 4).

## References

- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): `record_shapes`, `key_averages(group_by_input_shape=True)`, and `export_chrome_trace`.
- [Generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation): the cache API used for decode.

## Practice

Complete [exercise.py](exercise.py):

1. `run_prefill()` and `run_decode()`: perform a warmed-up prefill of about 256 tokens and 16 cached decode steps that reuse a precomputed cache.
2. `profile_phase()`: profile one phase with CPU and CUDA activities and `record_shapes=True`, wrapped in `record_function(label)`.
3. `top_table()`: produce the top operators by self device time (self CPU time on CPU), grouped by input shape.
4. The script exports traces to `results/<phase>.pt.trace.json`, which git ignores because traces are large, and saves the tables to `results/<phase>_top_ops.txt`.

Then open both traces and write down at least two differences you observe, using operator shapes or execution behavior.

```bash
python week_03_gpu_execution_and_measurement/day_17_profile_prefill_and_decode/exercise.py
```

## Done when

You have a profile for both phases and can explain at least two observed differences using operator shapes or execution behavior.
