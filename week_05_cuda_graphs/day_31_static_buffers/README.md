# Day 31: Static Buffers and Correctness

**Time:** 60 minutes. **Goal:** Feed new values through captured storage safely.

## Knowledge

A graph captures the addresses used by GPU operations, not a Python variable's
future binding. Assigning `case.x = new_tensor` changes a Python reference but
does not redirect the captured graph. Copying new values into the original input
storage preserves the address the graph uses. Keep input, temporary, and output
buffers alive and avoid resizing them between capture and replay.

The output buffer is reused too. If you need to retain a result across another
replay, clone it before the next execution overwrites that storage. An apparently
perfect comparison can otherwise compare two aliases of the same updated result.
The solution clones the eager reference before replay and uses `assert_close`
on changed inputs rather than merely checking one initial capture.

Input transfers or buffer updates are real application costs. Today's correctness
test includes updates, but Week 5's steady-state microbenchmark excludes them from
its timing boundary. Report this distinction explicitly. A production graph pipeline
must handle stream ordering between input copies and graph launches; persistent
addresses alone do not guarantee that new values are ready to read.

## References

- [PyTorch graph constraints](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs): long-lived storage.
- [Tensor comparison](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close): numerical checks.

## Practice

Complete [exercise.py](exercise.py). Use negative, zero, and positive inputs, and
verify the input pointer stays unchanged. Explain why cloning the expected output
is necessary. Compare with [solution.py](solution.py).

## Run

```bash
python week_05_cuda_graphs/day_31_static_buffers/exercise.py
python week_05_cuda_graphs/day_31_static_buffers/solution.py
```

**Done when:** All changed inputs pass and `results/correctness.json` records actual errors.
