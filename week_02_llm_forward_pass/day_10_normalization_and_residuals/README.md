# Day 10: Normalization and Residual Connections

**Time:** 60 minutes

## Goal

Understand how normalization and residual additions fit into a decoder block.

## Knowledge

**RMSNorm** rescales each token's hidden vector to unit root mean square, then applies a learned per-channel scale `g`:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

**LayerNorm** first subtracts the mean, then divides by the standard deviation, and typically adds a learned bias. RMSNorm skips the mean subtraction and the bias. It costs one reduction instead of two and works just as well in practice for LLMs. You will check one consequence directly: adding a constant to every channel changes the RMSNorm output but not the LayerNorm output.

`ε` (`rms_norm_eps` in the config) prevents division by zero. The Hugging Face Qwen3 RMSNorm computes in FP32 internally and casts back to the input dtype. That choice protects accuracy in BF16.

**The pre-norm residual block** used by Qwen3:

```text
h   = x + Attention(RMSNorm_1(x))        # residual add 1
out = h + MLP(RMSNorm_2(h))              # residual add 2
```

Every tensor above has the shape `[B, T, hidden]`. The residual stream `x → h → out` passes through the whole model unchanged in shape, and each block *adds* a correction to it. Normalization happens *before* each sublayer (pre-norm), and a final `model.norm` runs after the last block. Qwen3 also applies a small RMSNorm to each query and key head (`q_norm`, `k_norm`, over `head_dim`) before RoPE.

**Operation types and their GPU implications:**

| Operation | Type | Notes |
| --- | --- | --- |
| Mean of `x²` across `hidden` | Reduction | Needs a cross-thread sum for each token |
| `x * rsqrt(...)`, `* g` | Elementwise | Memory-bound: reads and writes the whole tensor |
| Residual add | Elementwise | Memory-bound |

Unfused, "residual add, then RMSNorm" reads and writes the hidden tensor several times. A fused kernel could read it once and write it once. That is a **hypothesis** about saving memory traffic. Week 3 shows how to test it, and the Day 28 next project could implement it.

## References

- [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html): the formula and the `eps` argument.
- [Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py): `Qwen3RMSNorm` and `Qwen3DecoderLayer.forward`. Read the version you have installed, found with `python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"`.
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): the residual connections section.

## Practice

Complete [exercise.py](exercise.py):

1. `rms_norm()`: implement the formula and compare it with `torch.nn.RMSNorm` and the Hugging Face `Qwen3RMSNorm`.
2. `layer_norm_vs_rms_norm()`: show that a constant offset changes the RMSNorm output but not the LayerNorm output.
3. `trace_block_shapes()`: register forward hooks on the real model's block 0 (`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`, and the block itself) and record each output shape. Verify the block equation by recomputing `out` from the hooked parts.

```bash
python week_02_llm_forward_pass/day_10_normalization_and_residuals/exercise.py
```

## Done when

You can explain normalization and residuals and identify their locations in the real block.
