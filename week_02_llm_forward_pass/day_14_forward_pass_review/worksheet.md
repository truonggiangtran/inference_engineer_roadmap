# Week 2 Review Worksheet

Answer from memory first. Mark any answer you had to look up.

## 1. Annotated forward pass

Fill in the shapes using your Day 8 values (`B=1`, `T=` your prompt length).

| Stage | Shape | Learned parameters used | Model-specific? |
| --- | --- | --- | --- |
| Token IDs | | | |
| Embeddings | | | |
| `q_proj` output | | | |
| `k_proj` / `v_proj` output | | | |
| After split heads (Q) | | | |
| After split heads (K/V) | | | |
| Attention scores | | | |
| `o_proj` output | | | |
| MLP intermediate | | | |
| Block output | | | |
| Logits | | | |

## 2. Prefill compared with one cached decode step

For the same prompt of `P` tokens, describe each phase:

| | Prefill | One cached decode step |
| --- | --- | --- |
| Input tokens | | |
| Q shape (one layer) | | |
| K/V read from the cache | | |
| K/V written to the cache | | |
| Score matrix shape | | |
| Logits needed | | |

## 3. Parameters, activations, and cache

Classify each item as learned parameter, intermediate activation, or cached state: `q_proj.weight`, attention scores, layer-5 keys after RoPE, MLP `gate` output, final norm scale, the hidden state after block 3, and `lm_head.weight`.

> 

## 4. Review questions

1. What does one decoder block compute, and which parts are specific to the chosen model?

   > 

2. How do RoPE, residuals, normalization, and the MLP fit into the forward pass?

   > 

3. What separates logits, probabilities, and token selection?

   > 

4. How do pretraining, instruction tuning, and inference differ?

   > 

5. What changes between prefill, uncached generation, and cached decoding?

   > 

6. Why does KV caching avoid recomputation without eliminating attention over the existing context?

   > 

## 5. Ten-minute explanation outline

- 

## 6. Open questions for Week 3

- 
