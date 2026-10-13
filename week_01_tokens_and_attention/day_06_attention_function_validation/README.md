# Day 6: Build and Validate a Small Attention Function

**Time:** 90 minutes

## Goal

Turn the previous exercises into one understandable, testable forward operation.

## Knowledge

**A reference implementation is a test oracle.** `torch.nn.functional.scaled_dot_product_attention` (SDPA) computes the same mathematics as your Day 5 code, possibly with a fused kernel such as FlashAttention or memory-efficient attention. If your function agrees with SDPA on random inputs, you have strong evidence that the shapes, scaling, mask, and softmax axis are all correct.

**Comparing floating-point results.** Two correct implementations rarely match bit for bit, because they can sum terms in a different order. `torch.testing.assert_close(actual, expected, rtol, atol)` passes when, element by element:

$$
|a - e| \le \text{atol} + \text{rtol} \cdot |e|
$$

Choose tolerances based on the dtype and the size of the computation, and write down your reasoning:

- **FP32:** machine epsilon is about `1.2e-7`. For short reductions like these, errors around `1e-6` are expected. `assert_close` defaults to `rtol=1.3e-6` and `atol=1e-5` for float32.
- **BF16 or FP16:** errors are much larger, around `1e-2`. Never reuse FP32 tolerances for them.

If a check fails, investigate before loosening the tolerance. A wrong axis typically produces errors of order 1, not `1e-6`.

**Causal independence test.** Causality is a functional property: the output at position `i` must not depend on the inputs at positions after `i`. To test it, perturb the query, key, and value at the last position and check that outputs `0..T-2` are unchanged. This catches a missing or misaligned mask, which other checks can miss.

**SDPA arguments used here:** `is_causal=True` with equal query and key lengths, and `dropout_p=0.0` (dropout is a training feature). Day 13 explains why `is_causal` is not the right mask when a single query attends to a longer cached key sequence.

**Python modules.** Putting the logic in a function with explicit inputs and outputs makes it importable (`from solution import causal_attention`) and testable.

## References

- [Scaled dot-product attention API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): shape conventions, `is_causal`, and `attn_mask`.
- [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close): tolerances and default values.
- [Python modules](https://docs.python.org/3/tutorial/modules.html): imports and `if __name__ == "__main__"`.

## Practice

Complete [exercise.py](exercise.py):

1. `causal_attention(q, k, v)`: a single documented function that returns the output `[B, H, T, D]`.
2. `compare_with_sdpa()`: compare your function with SDPA in FP32 using explicit tolerances, and return the maximum absolute error.
3. `causal_independence()`: change only the last token's `q`, `k`, and `v`, then verify that the earlier outputs are unchanged and the last output does change.
4. Run both checks for sequence lengths 4 and 7. The script saves `results/attention_validation.json` with a shape table and the errors.

```bash
python week_01_tokens_and_attention/day_06_attention_function_validation/exercise.py
```

## Done when

Your manual attention agrees with the reference within justified tolerances and passes the causal-independence check.
