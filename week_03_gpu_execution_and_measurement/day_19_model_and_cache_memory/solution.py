"""Day 19 solution: estimate and measure parameter, KV cache, and peak GPU memory."""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
PROMPT_TOKENS, DECODE_STEPS = 512, 32
MIB = 1024**2


def layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def parameter_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def kv_bytes_estimate(config, batch: int, tokens: int, bytes_per_element: int) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * batch * tokens * config.num_key_value_heads * head_dim * bytes_per_element


def cache_bytes(cache, num_layers: int) -> int:
    total = 0
    for layer in range(num_layers):
        keys, values = layer_kv(cache, layer)
        total += keys.numel() * keys.element_size() + values.numel() * values.element_size()
    return total


def measure_request(model, prompt_ids: torch.Tensor, decode_steps: int, use_cache: bool) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    cache, sequence = None, prompt_ids
    if use_cache:
        out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(decode_steps):
            mask = torch.ones(1, cache.get_seq_length() + 1, dtype=torch.long, device=DEVICE)
            out = model(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    else:
        for _ in range(decode_steps + 1):
            out = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
            sequence = torch.cat([sequence, out.logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize()

    return {
        "baseline_mib": baseline / MIB,
        "peak_delta_mib": (torch.cuda.max_memory_allocated() - baseline) / MIB,
        "retained_delta_mib": (torch.cuda.memory_allocated() - baseline) / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "cache_mib": cache_bytes(cache, model.config.num_hidden_layers) / MIB if cache is not None else 0.0,
        "cache_len": cache.get_seq_length() if cache is not None else None,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    config = model.config
    bpe = torch.tensor([], dtype=DTYPE).element_size()

    final_tokens = PROMPT_TOKENS + DECODE_STEPS
    report = {
        "device": DEVICE, "dtype": str(DTYPE), "prompt_tokens": PROMPT_TOKENS, "decode_steps": DECODE_STEPS,
        "parameter_mib": parameter_bytes(model) / MIB,
        "kv_estimate_mib": kv_bytes_estimate(config, 1, final_tokens, bpe) / MIB,
        "kv_estimate_per_token_kib": kv_bytes_estimate(config, 1, 1, bpe) / 1024,
    }
    print(json.dumps(report, indent=2))

    if DEVICE == "cuda":
        ids = tokenizer("Memory accounting separates retained storage from temporaries. " * 80)["input_ids"]
        prompt_ids = torch.tensor([ids[:PROMPT_TOKENS]], device=DEVICE)
        report["model_allocated_mib"] = torch.cuda.memory_allocated() / MIB
        with torch.inference_mode():
            report["cached"] = measure_request(model, prompt_ids, DECODE_STEPS, use_cache=True)
            report["uncached"] = measure_request(model, prompt_ids, DECODE_STEPS, use_cache=False)
        assert report["cached"]["cache_len"] == final_tokens
        error = report["cached"]["cache_mib"] / report["kv_estimate_mib"] - 1
        print(json.dumps({k: report[k] for k in ("model_allocated_mib", "cached", "uncached")}, indent=2))
        print(f"measured cache vs estimate: {error:+.2%}")
    else:
        print("CPU only: CUDA allocator statistics unavailable; estimates only.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "memory.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
