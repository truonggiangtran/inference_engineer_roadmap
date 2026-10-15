# Day 13: Prefill, Cached Decoding, and Cache Correctness

**Time:** 90 minutes

## Goal

Understand what caching reuses and how a cached forward pass differs from recomputing the prefix.

## Knowledge

**The redundancy in Day 12's loop.** In a decoder, the key and value of token `j` depend only on tokens `0..j`, because of the causal mask. Once computed, they never change. Recomputing them at every step is wasted work.

**The KV cache** stores each layer's K and V after RoPE has been applied:

```text
per layer:  keys, values: [B, H_kv, T_cached, D]
```

**Two phases:**

| Phase | Input | Work | Output |
| --- | --- | --- | --- |
| **Prefill** | All `P` prompt tokens | Full forward; fills the cache with `P` positions | Logits for the next token |
| **Decode step** | **One** new token | Q, K, V for that token only; **appends** K/V to the cache; the new query attends to **all** cached keys | Logits for the following token |

What is reused, and what still happens in a decode step:

- **Reused:** the previous keys and values.
- **Still computed:** the new token's query, key, and value, plus its MLP and the other per-token work, because they depend on the new token.
- **Still read:** the whole valid cache. Attention over existing context is not eliminated; it becomes a `[1 × T_cached]` score row instead of a full `[T × T]` matrix.

Decode cost per step therefore grows with context length, mostly as memory reads.

**Masks and positions with a cache.** A single query against `T_cached + 1` keys is *not* a square causal problem. The new token may attend to every cached position. Passing `is_causal=True` to SDPA with `T_q = 1` and `T_k > 1` would align the mask to the top-left, wrongly hiding almost all keys. Let the model's public API handle the alignment:

- Pass `past_key_values=cache` and `use_cache=True`.
- Pass an `attention_mask` of ones covering **past plus current** tokens, shaped `[B, T_cached + 1]`.
- The model derives the new token's position (for RoPE) from the cache length.

**Request isolation.** A cache belongs to one request. Reusing it for an unrelated prompt silently corrupts the results, so start each independent request with a fresh cache.

**Correctness before speed.** Cached and uncached runs should produce the same last-position logits for the same tokens, up to floating-point reduction differences. Check this in FP32 first.

## References

- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): `DynamicCache` and how the cache grows.
- [Cache strategies](https://huggingface.co/docs/transformers/kv_cache): the public cache API. Advanced cache types are optional.
- [Generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation): `use_cache` and `past_key_values`.

## Practice

Complete [exercise.py](exercise.py). It uses FP32 for the numerical comparison.

1. `prefill()`: forward the prompt with `use_cache=True` and return the last-position logits and the cache.
2. `decode_step()`: forward **one** token with the cache and a mask covering past and current tokens.
3. `layer_kv()` and `cache_shapes()`: read the per-layer key and value shapes. The helper handles both Hugging Face cache layouts.
4. The script runs eight greedy decode steps. At each step it checks that the cache length grew by one, then compares the cached logits with an uncached full-prefix forward and records the maximum absolute and relative errors.

```bash
python week_02_llm_forward_pass/day_13_prefill_and_cached_decoding/exercise.py
```

## Done when

You can explain why previous keys and values are retained, why the next query is still computed, and why cached decoding still reads the valid context.
