"""Day 13 exercise: prefill, cached decoding, and cached-vs-uncached correctness."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
DECODE_STEPS = 8
# Starting FP32 tolerances for logits after 28 layers; justify or tighten them from your data.
RTOL, ATOL = 1e-3, 1e-3


def layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keys, values) for one layer across Hugging Face cache layouts."""
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def prefill(model, prompt_ids: torch.Tensor):
    """TODO: forward the whole prompt with use_cache=True and an all-ones attention_mask.
    Return (last_position_logits [1, vocab], cache) where cache = outputs.past_key_values."""
    raise NotImplementedError("TODO: implement prefill")


def decode_step(model, token_id: torch.Tensor, cache):
    """TODO: forward ONE token ([1, 1]) with past_key_values=cache, use_cache=True, and
    attention_mask=ones(1, cache.get_seq_length() + 1). Return (logits [1, vocab], cache)."""
    raise NotImplementedError("TODO: implement decode_step")


def cache_shapes(cache, num_layers: int) -> list[list[int]]:
    """TODO: return [list(keys.shape)] for every layer using layer_kv; also assert keys and values match."""
    raise NotImplementedError("TODO: implement cache_shapes")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    prompt_ids = tokenizer("A KV cache stores keys and values so that", return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = prompt_ids.shape[1]
    records = []

    with torch.inference_mode():
        logits, cache = prefill(model, prompt_ids)
        assert cache.get_seq_length() == prompt_len
        shapes = cache_shapes(cache, config.num_hidden_layers)
        expected = [1, config.num_key_value_heads, prompt_len, head_dim]
        assert all(s == expected for s in shapes), f"expected every layer {expected}, got {shapes[0]}"
        print(f"after prefill: {len(shapes)} layers x keys {shapes[0]} (and values)")

        sequence = prompt_ids
        for step in range(DECODE_STEPS):
            next_id = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            sequence = torch.cat([sequence, next_id], dim=1)

            logits, cache = decode_step(model, next_id, cache)
            assert cache.get_seq_length() == prompt_len + step + 1, "cache must grow by exactly one"

            uncached = model(input_ids=sequence, use_cache=False).logits[:, -1]
            abs_err = (logits - uncached).abs().max().item()
            rel_err = ((logits - uncached).abs() / uncached.abs().clamp_min(1e-6)).max().item()
            torch.testing.assert_close(logits, uncached, rtol=RTOL, atol=ATOL)
            same_argmax = bool(logits.argmax() == uncached.argmax())
            records.append({"step": step, "cache_len": cache.get_seq_length(), "max_abs_err": abs_err,
                            "max_rel_err": rel_err, "same_argmax": same_argmax})
            print(records[-1])

    print(f"generated: {tokenizer.decode(sequence[0, prompt_len:])!r}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "cache_correctness.json").write_text(json.dumps(
        {"dtype": str(DTYPE), "rtol": RTOL, "atol": ATOL, "prompt_len": prompt_len, "steps": records}, indent=2
    ))
    print("All Day 13 checks passed.")


if __name__ == "__main__":
    main()
