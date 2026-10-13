# Day 8: Inspect a Real Small Decoder-Only Model

**Time:** 60 minutes

## Goal

Connect the toy operations to a pretrained model's actual configuration and modules.

## Knowledge

**Configuration fields.** `model.config` holds the architecture hyperparameters. The following are the ones you'll use all month. Record the actual values from your run; this README deliberately leaves them blank.

| Field | Meaning |
| --- | --- |
| `num_hidden_layers` | Number of repeated decoder blocks |
| `hidden_size` | Width of the residual stream (the vector per token between blocks) |
| `intermediate_size` | Expanded width inside the MLP |
| `vocab_size` | Rows in the embedding table and columns of the output logits |
| `num_attention_heads` | Query heads |
| `num_key_value_heads` | Key/value heads. Fewer than query heads means grouped-query attention (Day 9). |
| `head_dim` | Per-head width. **It may be set explicitly**, and then `num_attention_heads * head_dim` can differ from `hidden_size`. |

Only derive `head_dim = hidden_size // num_attention_heads` when the config doesn't specify it. Check the projection weight shapes to confirm which value the model actually uses:

```text
q_proj.weight: [num_attention_heads * head_dim, hidden_size]
k_proj.weight: [num_key_value_heads * head_dim, hidden_size]
o_proj.weight: [hidden_size, num_attention_heads * head_dim]
```

**Module tree of a Hugging Face causal LM such as Qwen3:**

```text
Qwen3ForCausalLM
├── model (Qwen3Model)
│   ├── embed_tokens          embedding table
│   ├── layers[i] (decoder block) x N
│   │   ├── input_layernorm   RMSNorm before attention
│   │   ├── self_attn         q_proj, k_proj, v_proj, q_norm, k_norm, o_proj
│   │   ├── post_attention_layernorm   RMSNorm before the MLP
│   │   └── mlp               gate_proj, up_proj, down_proj
│   ├── norm                  final RMSNorm
│   └── rotary_emb            RoPE cos/sin generator (Day 9)
└── lm_head                   hidden -> vocabulary logits (may share weights with embed_tokens)
```

**Choosing a dtype.** Load the weights in one explicitly chosen dtype. BF16 halves memory compared with FP32 and is the usual inference choice on modern GPUs; FP32 is useful for strict numerical checks. Record the choice, because it affects both speed and numerical behavior.

**Thinking mode.** Qwen3 can emit `<think>` reasoning blocks. Use `enable_thinking=False` in the chat template so every run uses the same output format.

## References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): the loading example and the thinking or non-thinking switch.
- [Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3): the configuration class.
- [Qwen3-0.6B `config.json`](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json): the raw configuration fields.

## Practice

Complete [exercise.py](exercise.py). It downloads about 1.5 GB of weights the first time.

1. `config_summary()`: extract the fields above, and prefer an explicit `head_dim` when the config has one.
2. `locate_components()`: return the embedding, one decoder block, its attention, MLP, and norms, the final norm, and `lm_head`.
3. The script checks the projection weight shapes against your summary and reports the parameter count and whether the embedding and output weights are tied.
4. `generate_short()`: run a short greedy generation using the chat template with thinking disabled.

```bash
python week_02_llm_forward_pass/day_08_inspect_decoder_model/exercise.py
```

The script saves `results/model_anatomy.json` and `results/module_tree.txt`. Copy the numbers into the Model Anatomy worksheet in the roadmap.

## Done when

You can locate the components learned in Week 1 inside the real model and explain why its query and KV head counts may differ.
