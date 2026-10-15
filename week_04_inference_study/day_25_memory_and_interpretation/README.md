# Day 25: Collect Memory and Explain the Results

**Time:** 60 minutes

## Goal

Relate observed latency and memory differences to cached computation and context length.

## Knowledge

**Memory measurement protocol, for each of the four conditions:**

1. Release the previous request's outputs and cache (`del`, then `gc.collect()`), and call `torch.cuda.empty_cache()`.
2. Record `baseline = memory_allocated()`. At this point the model weights are already resident.
3. Call `reset_peak_memory_stats()`.
4. Run one full request: prefill plus 32 steps, both inside the measured region.
5. Record `peak_delta = max_memory_allocated() - baseline`, and `retained_delta = memory_allocated() - baseline` **while the cache is still alive**.
6. For the cached mode, sum the actual cache tensors and compare the total with the analytical KV estimate for `P + 32` tokens.

`memory_reserved()` can stay high because the allocator reuses blocks. That is not a leak; report it separately.

**Interpreting the results:**

- **Retained:** cached mode keeps roughly the KV estimate, and uncached mode keeps about 0.
- **Peak:** this includes temporary activations such as MLP intermediates `[T, intermediate]`, attention workspaces, and logits. Uncached steps re-run the full prefix, so their temporary activations scale with `T` at every step. Cached decode steps have `T = 1`. Either mode may have the larger peak, depending on the prompt length. Explain whichever you observe.
- **Discrepancies:** allocator rounding (blocks in multiples of 512 bytes, or 2 MiB segments), cache preallocation strategies, and workspaces make measured values differ from formulas. Explain the difference; don't force the data to match.

**Diagnostic traces explain timing; they don't produce timing.** Profile one cached and one uncached decode step at `P = 512` in **separate** runs from the timing ones. Connect speed differences to evidence, such as:

- Repeated projection and MLP GEMMs over the whole prefix in uncached mode.
- Attention shapes: `[T × T]` compared with `[1 × T]`.
- Cache reads.
- Allocation behavior.
- Host overhead.

If the evidence is inconclusive, say so.

## References

- [CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management): allocated compared with reserved memory, and peak statistics.
- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): cache storage.
- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): diagnostic traces and `profile_memory`.

## Practice

Complete [exercise.py](exercise.py):

1. `measure_memory()`: implement the protocol above for one condition, and return the values in MiB.
2. `diagnostic_profile()`: profile a single decode step (cached and uncached) at `P = 512`. Save the top-operator table to `results/` and export the trace, which git ignores.
3. The script saves `results/memory.json`. Then write your interpretation in the Day 27 report, including at least one case where a simple expectation was incomplete, or where the evidence remained inconclusive.

```bash
python week_04_inference_study/day_25_memory_and_interpretation/exercise.py
```

Memory statistics require CUDA. On CPU, the script records only the analytical estimates.

## Done when

You can explain both time and memory results, including a case where a simple expectation was incomplete or the evidence remained inconclusive.
