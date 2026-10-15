# Day 9: Position Information, RoPE, and Grouped-Query Attention

**Time:** 60 minutes

## Goal

Understand how attention represents token positions and why fewer KV heads can reduce cache storage.

## Knowledge

**Attention alone ignores order.** Without position information, `QK^T` gives the same scores if the tokens are permuted. Each model handles position differently:

- **GPT-2** adds a *learned positional embedding* (one vector per absolute position) to the token embedding.
- **Qwen3**, like Llama and many modern decoders, uses **rotary position embedding (RoPE)**. Positions are not extra vocabulary tokens and are not added to the hidden state. Instead, query and key vectors are *rotated* by position-dependent angles right before the dot product.

**RoPE mechanics.** Split each `D`-dimensional head vector into `D/2` pairs. Pair `i` at position `p` is rotated by the angle `p · θ_i`, where `θ_i = base^(-2i/D)` and `base` is `rope_theta` from the config. Hugging Face implements this as:

```text
cos, sin:   [T, D]   (each of the D/2 frequencies appears twice)
rope(x) = x * cos + rotate_half(x) * sin,   rotate_half([x1, x2]) = [-x2, x1]
```

Two properties you will test:

1. **Rotation preserves vector length.** `|rope(x)| = |x|`.
2. **Relative position.** `rope(q, m) · rope(k, n)` depends only on `m − n`. Shifting both positions by `s` leaves the score unchanged. This is why RoPE encodes *relative* distance inside attention.

RoPE is applied to queries and keys only, never to values. In cached decoding, keys are stored *after* rotation, so the cache already contains their position information.

**Grouped-query attention (GQA).** In grouped-query attention, `H_q` query heads share `H_kv < H_q` key/value heads. Each group of `n_rep = H_q / H_kv` consecutive query heads reads the same K and V:

```text
K, V: [B, H_kv, T, D] --repeat each KV head n_rep times--> [B, H_q, T, D]
query head h uses KV head h // n_rep
```

**Why GQA matters for inference.** The KV cache stores K and V for every layer and every past token, but only for the KV heads:

$$
\text{KV elements} = 2 \times \text{layers} \times B \times T \times H_{kv} \times D
$$

Using `H_q` instead of `H_kv` overestimates storage by a factor of `n_rep`. A smaller cache means less memory per sequence and fewer bytes to read in every decode step.

Not every decoder uses GQA: some use full multi-head attention (`H_kv = H_q`), and others use multi-query attention (`H_kv = 1`). Always read the config.

## References

- [RoFormer/RoPE paper](https://arxiv.org/abs/2104.09864): the introduction and section 3.2 intuition. The full derivation is optional.
- [Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3): the attention configuration fields.
- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): what the cache stores.

## Practice

Complete [exercise.py](exercise.py):

1. `build_rope_cache()` and `apply_rope()`: implement RoPE and verify the following:
   - It preserves length.
   - Position 0 leaves the vector unchanged.
   - The score is unchanged when both positions shift.
   - It matches the Hugging Face Qwen3 `apply_rotary_pos_emb`.
2. `repeat_kv()`: expand `[B, H_kv, T, D]` to `[B, H_q, T, D]`, and check that query head `h` sees KV head `h // n_rep`.
3. `kv_cache_elements()`: compute the cache size for a 128-token sequence using the real config (downloads only `config.json`), and compare the result with the query-head miscount.

```bash
python week_02_llm_forward_pass/day_09_rope_and_gqa/exercise.py
```

## Done when

You can explain position information and grouped-query attention at the shape and memory level without claiming every decoder uses the same architecture.
