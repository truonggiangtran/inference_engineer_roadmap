# Day 7: Explain Attention without Looking at the Code

**Time:** 45 minutes

## Goal

Consolidate the week and identify misunderstandings before adding the rest of the model.

## Knowledge

This week covered the whole causal attention path for one layer:

```text
text -> token IDs [B,T] -> embeddings [B,T,hidden]
     -> Q, K, V projections (GEMMs) [B,T,H*D] -> split heads [B,H,T,D]
     -> scores Q K^T / sqrt(D) [B,H,T,T] -> + causal mask -> softmax over keys
     -> weights @ V [B,H,T,D] -> merge heads [B,T,H*D] -> output projection [B,T,hidden]
```

The output projection (`o_proj`) maps the merged heads back to the hidden size. You will see it in the real model on Day 8.

Review means retrieving from memory, not rereading. Answer first, then check with the references or your Day 3–6 code only for questions you couldn't answer.

## References

Use these only for questions you could not answer from memory:

- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Practice

Fill in [worksheet.md](worksheet.md) without looking at your code, then compare with [solution.md](solution.md). Use any remaining time to finish a blocked exercise from Days 1–6.

## Done when

You can give a five-minute explanation of causal attention and answer the Week 1 review questions.
