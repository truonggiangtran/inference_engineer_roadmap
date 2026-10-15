# Day 11: MLP, Gating, Vocabulary Logits, and Token Choice

**Time:** 60 minutes

## Goal

Complete the computational path from a decoder block to the next-token prediction.

## Knowledge

**Gated MLP (SwiGLU).** Each block's MLP processes every token independently:

```text
x:     [B, T, hidden]
gate = gate_proj(x)              [B, T, intermediate]   GEMM (expand)
up   = up_proj(x)                [B, T, intermediate]   GEMM (expand)
act  = silu(gate) * up           [B, T, intermediate]   elementwise activation and gating
out  = down_proj(act)            [B, T, hidden]         GEMM (contract)
```

`silu(z) = z · sigmoid(z)`. The *gate* branch decides, per channel, how much of the *up* branch passes through. The three projections hold most of a block's parameters, so in a small dense model the MLP GEMMs are usually the largest share of the compute.

**From hidden states to logits.** After the last block and the final RMSNorm, `lm_head` maps each position's hidden vector to one score per vocabulary entry:

```text
final hidden [B, T, hidden] --lm_head--> logits [B, T, vocab]
```

Only the **last position's** logits are needed to predict the next token, which is why generation can request `logits_to_keep=1`.

**Logits are not probabilities.** They are unbounded real scores. `softmax` turns them into a distribution:

$$
p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

**Choosing a token:**

- **Greedy:** `argmax(logits)`. This is deterministic and unaffected by temperature, because dividing by a positive `τ` doesn't change which entry is largest.
- **Sampling with temperature `τ`:**
  - `τ < 1` sharpens the distribution, approaching greedy.
  - `τ > 1` flattens it, raising entropy.

Temperature changes only the *selection* step. The model weights and the logits themselves are unchanged.

**Four distinct objects:**

- **Hidden state:** the vector per token.
- **Logits:** scores per vocabulary entry.
- **Probabilities:** the logits after softmax.
- **Selected token ID:** a single integer that is appended to the sequence.

## References

- [Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py): `Qwen3MLP` and `Qwen3ForCausalLM.forward`.
- [Qwen3-0.6B `config.json`](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json): `hidden_act` and `intermediate_size`.
- [Generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation): `do_sample`, `temperature`, and greedy decoding.

## Practice

Complete [exercise.py](exercise.py):

1. `gated_mlp()`: implement the MLP and verify it against the Hugging Face `Qwen3MLP` built from a toy config.
2. `probabilities()` and `entropy()`: apply temperature to a tiny logits vector, then show that argmax is unchanged while entropy rises with `τ`.
3. `last_position_logits()`: run the real model, confirm that `lm_head(final hidden) == logits`, and print the five most likely next tokens with their probabilities.

```bash
python week_02_llm_forward_pass/day_11_mlp_logits_and_token_choice/exercise.py
```

## Done when

You can describe the complete block and distinguish hidden states, logits, probabilities, and selected token IDs.
