"""Day 8 exercise: inspect Qwen3-0.6B's configuration and modules."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"


def config_summary(config) -> dict:
    """TODO: return num_hidden_layers, hidden_size, intermediate_size, vocab_size,
    num_attention_heads, num_key_value_heads, head_dim, tie_word_embeddings, rope_theta,
    rms_norm_eps. Use config.head_dim when present, otherwise hidden_size // num_attention_heads,
    and record which in "head_dim_source"."""
    raise NotImplementedError("TODO: implement config_summary")


def locate_components(model) -> dict:
    """TODO: return a dict of modules with keys:
    embed_tokens, block0, attention, mlp, input_layernorm, post_attention_layernorm,
    final_norm, lm_head. Hint: print(model) and follow model.model.layers[0]..."""
    raise NotImplementedError("TODO: implement locate_components")


def generate_short(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
    """TODO: format [{"role": "user", "content": prompt}] with apply_chat_template(...,
    add_generation_prompt=True, enable_thinking=False), tokenize, move to the model's device,
    run model.generate(..., do_sample=False, max_new_tokens=...), and decode only the new tokens."""
    raise NotImplementedError("TODO: implement generate_short")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    print(f"device={DEVICE} dtype={DTYPE}")

    summary = config_summary(model.config)
    parts = locate_components(model)
    attn = parts["attention"]

    q_out = summary["num_attention_heads"] * summary["head_dim"]
    kv_out = summary["num_key_value_heads"] * summary["head_dim"]
    assert attn.q_proj.weight.shape == (q_out, summary["hidden_size"]), attn.q_proj.weight.shape
    assert attn.k_proj.weight.shape == (kv_out, summary["hidden_size"]), attn.k_proj.weight.shape
    assert attn.v_proj.weight.shape == (kv_out, summary["hidden_size"]), attn.v_proj.weight.shape
    assert attn.o_proj.weight.shape == (summary["hidden_size"], q_out), attn.o_proj.weight.shape
    assert parts["embed_tokens"].weight.shape[1] == summary["hidden_size"]

    summary["dtype"] = str(DTYPE)
    summary["device"] = DEVICE
    summary["revision"] = getattr(model.config, "_commit_hash", None)
    summary["num_parameters"] = sum(p.numel() for p in model.parameters())
    summary["embeddings_tied"] = parts["lm_head"].weight.data_ptr() == parts["embed_tokens"].weight.data_ptr()
    summary["weight_shapes_block0"] = {
        name: list(p.shape) for name, p in parts["block0"].named_parameters()
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "weight_shapes_block0"}, indent=2))
    print(f"query heads per KV head: {summary['num_attention_heads'] // summary['num_key_value_heads']}")

    answer = generate_short(model, tokenizer, "In one sentence, what is a KV cache?")
    print(f"\ngeneration: {answer!r}")
    summary["sample_generation"] = answer

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "model_anatomy.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "module_tree.txt").write_text(str(model))
    print(f"Saved {RESULTS_DIR}")


if __name__ == "__main__":
    main()
