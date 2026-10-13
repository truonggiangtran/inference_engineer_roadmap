# Day 5: Scaled Dot Products, Masking, and Softmax

**Time:** 60 minutes

## Goal

Understand the mathematical operations and numerical concerns of causal attention.

## Knowledge

For one head with `Q, K, V` shaped `[B, H, T, D]`:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{D}} + M\right) V
$$

| Step | Shape | Meaning |
| --- | --- | --- |
| `scores = Q @ K.transpose(-2, -1)` | `[B, H, T_q, T_k]` | Score of query `i` against key `j` |
| `/ sqrt(D)` | Same | Keeps scores from growing with `D` |
| `+ mask` | Same | Blocks future keys |
| `softmax(dim=-1)` | Same | Each *row* (one query) becomes a probability distribution over keys |
| `weights @ V` | `[B, H, T_q, D]` | Weighted average of value vectors |

**Why scale?** If the entries of `q` and `k` have unit variance, `q·k` has variance `D`. Large scores push softmax toward a one-hot output with vanishing gradients and make it sensitive to small changes. Dividing by `√D` restores roughly unit variance.

**Causal mask.** In a decoder, token `i` may only attend to keys `j ≤ i`. An additive mask puts `0` where attention is allowed and `-inf` above the diagonal. Because `exp(-inf) = 0`, future positions receive exactly zero weight. Every row still contains its own position, so no row is fully masked. A fully masked row would compute softmax over all `-inf` values and produce `NaN`.

**Softmax axis.** Softmax runs over the **key** dimension (`dim=-1`): "how much does this query attend to each key?" Normalizing over queries instead would be a bug that the checks catch, because rows would no longer sum to 1.

**Numerically stable softmax.** `exp(1000)` overflows. Softmax is unchanged when you subtract the same constant from every entry in a row, so stable implementations compute `exp(x - max(x))`. The largest exponent is then `exp(0) = 1`.

**Attention weights compared with attention output.** The weights `[T_q, T_k]` describe *where* each query looks. The output `[T_q, D]` is *what* it retrieves.

## References

- [Scaled dot-product attention API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): the reference formula and mask semantics.
- [`softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html): the `dim` argument.
- [Python control flow](https://docs.python.org/3/tutorial/controlflow.html): defining functions.

## Practice

Complete [exercise.py](exercise.py):

1. `causal_mask(T)`: return an additive `[T, T]` mask containing `0` and `-inf`.
2. `stable_softmax()`: implement softmax with row-maximum subtraction.
3. `attention_weights()`: compute the scaled scores, add the mask, and apply softmax over keys.
4. `attention_output()`: multiply the weights by `V`.

The checks confirm the following:

- Rows sum to 1.
- The upper triangle is zero.
- The stable softmax agrees with `torch.softmax`, including for values near 1000.
- The output matches a manual weighted sum.

```bash
python week_01_tokens_and_attention/day_05_scaled_dot_product_and_softmax/exercise.py
```

## Done when

You can explain scaling, the causal mask, and the softmax axis, and distinguish attention weights from the final attention output.
