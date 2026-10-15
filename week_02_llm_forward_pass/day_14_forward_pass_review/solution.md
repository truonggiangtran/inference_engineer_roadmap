# Week 2 Review: Model Answers

The shapes use symbols. Substitute the numbers you recorded on Day 8:

| Symbol | Config field |
| --- | --- |
| `L` | `num_hidden_layers` |
| `d` | `hidden_size` |
| `I` | `intermediate_size` |
| `V` | `vocab_size` |
| `Hq` | `num_attention_heads` |
| `Hkv` | `num_key_value_heads` |
| `D` | `head_dim` |

## 1. Annotated forward pass (`B=1`, prompt length `T`)

| Stage | Shape | Parameters | Model-specific? |
| --- | --- | --- | --- |
| Token IDs | `[1, T]` | None. The tokenizer is not a GPU layer. | Tokenizer and vocabulary are |
| Embeddings | `[1, T, d]` | `embed_tokens [V, d]` | — |
| `q_proj` output | `[1, T, Hq·D]` | `q_proj [Hq·D, d]` | `Hq·D` may differ from `d` (explicit `head_dim`) |
| `k_proj` / `v_proj` output | `[1, T, Hkv·D]` | `[Hkv·D, d]` each | GQA |
| Q after split | `[1, Hq, T, D]` | `q_norm [D]`, then RoPE | q/k norm and RoPE |
| K/V after split | `[1, Hkv, T, D]` | `k_norm [D]`, RoPE on K only | GQA, RoPE |
| Scores | `[1, Hq, T, T]` | — (each KV head is shared by `Hq/Hkv` query heads) | — |
| `o_proj` output | `[1, T, d]` | `o_proj [d, Hq·D]` | — |
| MLP intermediate | `[1, T, I]` | `gate_proj`, `up_proj [I, d]` | SwiGLU |
| Block output | `[1, T, d]` | `down_proj [d, I]`, two RMSNorm scales `[d]` | Pre-norm RMSNorm |
| Logits | `[1, T, V]` (or `[1, 1, V]` with `logits_to_keep=1`) | `lm_head [V, d]`, possibly tied | Tying is model-specific |

## 2. Prefill compared with one cached decode step

| | Prefill | Cached decode step |
| --- | --- | --- |
| Input tokens | `P` | 1 |
| Q shape | `[1, Hq, P, D]` | `[1, Hq, 1, D]` |
| K/V read from cache | None (empty) | `P + k` past positions, every layer |
| K/V written | `P` positions per layer | 1 position per layer |
| Scores | `[1, Hq, P, P]`, causal | `[1, Hq, 1, P+k+1]`. No future positions exist, so nothing needs masking. |
| Logits needed | Last position only | The single position |

## 3. Classification

- **Learned parameters:** `q_proj.weight`, the final norm scale, `lm_head.weight`.
- **Intermediate activations:** attention scores, the MLP `gate` output, the hidden state after block 3.
- **Cached state:** layer-5 keys after RoPE.

## 4. Review questions

1. **One block** computes `h = x + Attn(RMSNorm(x))`, then `out = h + MLP(RMSNorm(h))`. The following are choices of this model rather than universal features: RoPE, GQA, explicit `head_dim`, `q_norm`/`k_norm`, RMSNorm instead of LayerNorm, and the SwiGLU MLP.
2. **How the pieces fit together:**
   - RoPE rotates Q and K by position just before the dot product.
   - The residuals carry the hidden stream unchanged in shape, and each sublayer adds a correction.
   - RMSNorm is applied before each sublayer and once at the end.
   - The MLP transforms each token independently, expanding to `I` and contracting back to `d`.
3. **Logits, probabilities, and token selection:**
   - Logits are unbounded scores from `lm_head`.
   - Probabilities are `softmax(logits / τ)`.
   - Token selection picks an ID, either by argmax (greedy) or by sampling.
   - Temperature affects only the probabilities and the selection, never the weights.
4. **Pretraining, instruction tuning, and inference:**
   - Pretraining learns all weights from raw text through next-token prediction with shifted targets.
   - Instruction tuning further adjusts the weights on formatted conversations.
   - Inference keeps the weights fixed and only extends the sequence.
5. **Prefill, uncached generation, and cached decoding:**
   - Prefill processes the whole prompt once.
   - Uncached generation reruns the full, growing prefix at every step.
   - Cached decoding processes one new token per step, reusing the stored K/V for the prefix and appending its own.
6. **Why caching still reads the context:** past K/V never change under causal masking, so they can be reused instead of recomputed. The new query still needs a score against every past key and a weighted sum over every past value, so each step reads the whole valid cache. Its cost grows with context length.
