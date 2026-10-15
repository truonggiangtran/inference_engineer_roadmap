# Day 4: Q/K/V Projections and Attention Heads

**Time:** 60 minutes

## Goal

Connect your GEMM knowledge to the learned projections inside attention.

## Knowledge

**Projections are GEMMs.** Attention starts with three learned linear maps applied to every token's hidden vector:

```text
x: [B, T, hidden]
Q = x @ W_q^T     K = x @ W_k^T     V = x @ W_v^T
```

`nn.Linear(in, out)` stores `weight` as `[out, in]` and computes `x @ weight.T`. For the GEMM, flatten the batch and sequence dimensions: `M = B*T` rows, `K = hidden`, and `N = out_features`. Prefill has a large `M` (many tokens), while one decode step has `M = B` (one token per sequence). That difference shapes Week 3.

**Roles:**

- **Query:** what this token is looking for.
- **Key:** what this token offers to be matched against.
- **Value:** the information this token passes along when it is attended to.

**Multi-head layout.** With `H` heads of size `D` (`H*D = out_features`), the last dimension is split into heads, and the heads axis is moved in front of the tokens axis. This lets each head attend over tokens independently, as a batched matrix multiplication:

```text
[B, T, H*D] --view--> [B, T, H, D] --transpose(1, 2)--> [B, H, T, D]
                                                          ^  ^
                                                       heads tokens
```

**Views, strides, and contiguity.** A tensor is a pointer to storage plus *strides*, the number of elements to step to move one index along each dimension. Both `view` and `transpose` return new views of the *same storage*. They only change the strides, so no data is copied:

| Step | Shape | Strides |
| --- | --- | --- |
| `x` | `[2, 4, 8]` | `(32, 8, 1)` |
| `.view(2, 4, 2, 4)` | `[2, 4, 2, 4]` | `(32, 8, 4, 1)` |
| `.transpose(1, 2)` | `[2, 2, 4, 4]` | `(32, 4, 8, 1)`: not contiguous |

A tensor is *contiguous* when its elements are laid out in row-major order with no gaps. Some operations, such as `view` on a non-contiguous tensor and some kernels, require that layout, so `.contiguous()` makes a copy. That copy is real memory traffic on the GPU and shows up in profiles (Week 3).

## References

- [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html): batched multiplication semantics.
- [PyTorch tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html): reshaping and transposing.
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): the self-attention projection and head-splitting sections.

## Practice

Complete [exercise.py](exercise.py):

1. `project_qkv()`: apply three bias-free `nn.Linear(8, 8)` projections to `x` shaped `[2, 4, 8]`.
2. `split_heads()`: reshape `[B, T, H*D]` into `[B, H, T, D]` with `H=2`, `D=4`.
3. `merge_heads()`: invert `split_heads()`.
4. Use `describe_layout()` to print shapes, strides, contiguity, and storage pointers before and after transposing.

The checks confirm the following:

- Each head is the correct slice of the projection.
- The round trip through `split_heads()` and `merge_heads()` is exact.
- Flattening the transposed tensor with `view(-1)` fails until you make it contiguous.

```bash
python week_01_tokens_and_attention/day_04_qkv_projections_and_heads/exercise.py
```

## Done when

You can connect each projection to a matrix multiplication and explain the layout transformation without confusing token and head axes.
