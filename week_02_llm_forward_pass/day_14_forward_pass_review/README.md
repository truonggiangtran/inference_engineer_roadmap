# Day 14: Explain the Full LLM Forward Pass

**Time:** 45 minutes

## Goal

Produce an architecture explanation that connects the model's meaning, tensor operations, and execution phases.

## Knowledge

You now have every piece of the forward pass:

```text
text --tokenizer--> ids [B,T]
     --embed_tokens--> x [B,T,hidden]
     --N x decoder block-->
         h   = x + o_proj(Attn(RoPE(q_norm(q_proj(norm1(x)))), RoPE(k_norm(k_proj(norm1(x)))), v_proj(norm1(x))))
         x'  = h + down_proj(silu(gate_proj(norm2(h))) * up_proj(norm2(h)))
     --final norm--> [B,T,hidden]
     --lm_head--> logits [B,T,vocab] --last position--> token choice
```

**Model-specific choices.** The following are Qwen3 choices, not universal LLM features:

- RoPE instead of learned absolute positions.
- Grouped-query attention with an explicit `head_dim`.
- Pre-norm RMSNorm.
- `q_norm` and `k_norm`.
- A SwiGLU gated MLP.
- Possibly tied embeddings: check the `embeddings_tied` value from your Day 8 results.

**Three kinds of state, which you should never mix up:**

| Kind | Examples | Lifetime |
| --- | --- | --- |
| Learned parameters | Projection weights, norm scales, embeddings | Fixed for every request |
| Intermediate activations | Hidden states, scores, MLP intermediates | Temporary, freed during or after the forward pass |
| Cached state | Per-layer K/V | Lives for one request and grows by one position per step |

## References

Use these only for unresolved details:

- [Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation)

## Practice

Fill in [worksheet.md](worksheet.md) from memory, then compare with [solution.md](solution.md). Use your Day 8 `model_anatomy.json` for the real numbers.

## Done when

You can give a ten-minute forward-pass explanation and answer the Week 2 review questions without copying the references.
