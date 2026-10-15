# Day 19: Account for Model and Cache Memory

**Time:** 60 minutes

## Goal

Separate retained model/cache storage from temporary allocations and allocator bookkeeping.

## Knowledge

**Four quantities that are easy to confuse:**

| Quantity | What it is | How to get it |
| --- | --- | --- |
| Parameter storage | Weights in their actual dtype | `sum(p.numel() * p.element_size())`. Tied weights are counted once. |
| KV cache storage | Retained K/V for one request | Formula below, or the sum of the cache tensors |
| Peak temporary usage | Activations, attention scores, MLP intermediates, logits | `torch.cuda.max_memory_allocated()` minus a baseline |
| Allocator reserve | Memory PyTorch's caching allocator holds for reuse | `torch.cuda.memory_reserved()` (≥ allocated) |

**KV cache formula** for a basic, unquantized, full-context cache:

$$
\text{KV bytes} = 2 \times L \times B \times T_{cached} \times H_{kv} \times D \times b
$$

It grows **linearly** with context length and batch size. This is logical tensor storage; it doesn't include allocator rounding or fragmentation.

**Allocated compared with reserved compared with `nvidia-smi`:**

- **`memory_allocated`:** bytes in live tensors.
- **`memory_reserved`:** allocated memory plus cached free blocks that PyTorch keeps instead of returning to the driver.
- **`nvidia-smi`:** everything on the device, including the reserved pool, the CUDA context, cuBLAS workspaces, other processes, and the display.

The three values will never match. Explain the differences instead of forcing them to agree.

**A peak measurement needs defined boundaries:**

1. Free the previous request's outputs (`del`, then `gc.collect()`).
2. Optionally call `torch.cuda.empty_cache()`.
3. Record the baseline allocation.
4. Call `torch.cuda.reset_peak_memory_stats()`.
5. Run the request.
6. Read `max_memory_allocated() - baseline`.

State whether prefill is inside the measured region. In this exercise it is.

**A counterintuitive result to watch for.** Cached decoding *retains* more memory, because the cache persists. Its **peak**, however, may be smaller than uncached execution's: every uncached step re-runs the full sequence and allocates full-length temporary activations. Retained storage and peak usage are different questions.

## References

- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): what the cache holds.
- [CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management): the caching allocator and `memory_allocated` compared with `memory_reserved`.
- [Qwen3-0.6B `config.json`](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json): the layer, KV-head, and `head_dim` values.

## Practice

Complete [exercise.py](exercise.py):

1. `parameter_bytes()`: compute the weight storage in its actual dtype.
2. `kv_bytes_estimate()`: implement the formula above, using the real config.
3. `cache_bytes()`: sum the actual cache tensors.
4. `measure_request()`: for a 512-token prompt with 32 decode steps, run the cached path and then the uncached path. Record the baseline, peak delta, retained delta, and reserved memory for each.
5. Compare the estimated cache size with the measured one. Write down why the peak delta is larger than the cache, and which mode has the larger peak.

```bash
python week_03_gpu_execution_and_measurement/day_19_model_and_cache_memory/exercise.py
```

On CPU, only the analytical estimates are produced. CUDA allocator statistics require a GPU.

## Done when

You can calculate approximate KV storage and explain why estimates, PyTorch memory statistics, and total GPU usage differ.
