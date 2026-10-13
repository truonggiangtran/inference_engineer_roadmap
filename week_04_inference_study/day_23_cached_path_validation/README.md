# Day 23: Validate the Cached Path before Timing It

**Time:** 60 minutes

## Goal

Establish that caching changes execution strategy without silently changing the intended computation.

## Knowledge

**What should match.** For the same prefix, the cached path's last-position logits should equal the uncached path's, up to floating-point differences. Those differences have a real cause: the same keys are produced by GEMMs with different `M`, such as a one-token decode compared with a full-prefix recompute. The kernels then tile and accumulate differently, so the rounding differs.

**Order of checks:**

1. **Short FP32 check (128-token workload).** FP32 rounding is small, so any structural bug stands out clearly. Examples of structural bugs:
   - A wrong mask.
   - Wrong positions or RoPE offsets.
   - A stale cache from a previous request.

   These produce errors of order 1, not `1e-4`.
2. **Benchmark dtype (BF16), both workloads.** BF16 keeps about 3 significant decimal digits (machine epsilon `2^-8 ≈ 0.004`). Across 28 layers, logit differences of order `0.1` are plausible. Choose tolerances using the logit scale, then *look at the distribution* of the errors.

**Metrics to record:**

- **Maximum absolute error** and **maximum relative error** for each step.
- **Argmax agreement.** It is informational only: matching token IDs alone don't prove the logits are close, and nearly tied top-two logits can flip the argmax even when the logits are correct. When the argmax differs, record the **top-1 versus top-2 margin**. A small margin means a benign tie; a large margin indicates a real bug.
- **Cache length after each step:** exactly `P + i + 1`.

**When a check fails, investigate before loosening tolerances.** Plot the error against the step number:

- Errors that grow with the step suggest an accumulation or mask problem.
- An error that is large on the first decode step suggests positions or the mask.
- An error spike on a single step usually comes from a near tie.

Speed and memory claims are only meaningful after this report passes.

## References

- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): cache initialization and updates.
- [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close): `rtol` and `atol`.
- [Cache strategies](https://huggingface.co/docs/transformers/kv_cache): the public API for your installed version.

## Practice

Complete [exercise.py](exercise.py):

1. `compare_logits()`: for two lists of per-step logits, return the per-step maximum absolute and relative errors, the argmax agreement, and the top-1 versus top-2 margins where the argmax differs. Also return whether every step passes `assert_close` at the given tolerances.
2. The script runs FP32 on the 128-token workload, then BF16 on both workloads, using fresh caches each time. It checks the cache-length growth and saves `results/cache_correctness.json`.

The tolerances in `TOLERANCES` are starting points. Justify or adjust them in the report from your observed error distribution, not to make a failure disappear.

```bash
python week_04_inference_study/day_23_cached_path_validation/exercise.py
```

## Done when

Both execution paths pass your documented correctness criteria for the controlled inputs before speed or memory claims are made.
