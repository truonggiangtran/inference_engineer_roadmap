# Day 33: Shape Buckets

**Time:** 60 minutes. **Goal:** Handle several supported shapes without unsafe replay.

## Knowledge

Captured operations normally use fixed shapes, layouts, and storage addresses.
A variable request length cannot simply be passed to an already captured graph
as a new Python argument. One strategy is to capture several supported shapes
and route each request to the smallest fitting bucket. The padded work and retained
buffers are costs, not free flexibility.

Today's pointwise operation has no interactions between vector elements, so
padding and trimming are straightforward. This does not establish correctness for
attention: padding masks, positional information, cache lengths, and reductions
must also be correct in a model. An unsupported length should raise an explicit
error rather than truncate data or pretend a bucket exists.

Retaining multiple graphs may retain private memory pools and buffers. Measure
allocated memory while all graph objects remain alive, and distinguish it from
reserved allocator memory and total device usage. The solution's allocated delta
includes persistent buffers, not merely graph metadata. A serving runtime may
employ more sophisticated capture sizes and fallback paths; this toy exercise
teaches the underlying tradeoff without claiming to reproduce their policies.

## References

- [PyTorch graphs and memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs): storage constraints and graph pools.

## Practice

Implement routing in [exercise.py](exercise.py). In [solution.py](solution.py),
inspect one retained graph per bucket. Check trimmed outputs and report padding
for three input lengths, then measure retained allocated storage.

## Run

```bash
python week_05_cuda_graphs/day_33_shape_buckets/exercise.py
python week_05_cuda_graphs/day_33_shape_buckets/solution.py
```

**Done when:** Unsupported lengths fail explicitly and `results/buckets.json` records padding and memory.
