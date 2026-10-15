"""Day 19 exercise: estimate and measure parameter, KV cache, and peak GPU memory."""

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
    """TODO: sum numel * element_size over model.parameters() (shared/tied weights appear once)."""
    raise NotImplementedError("TODO: implement parameter_bytes")


def kv_bytes_estimate(config, batch: int, tokens: int, bytes_per_element: int) -> int:
    """TODO: 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_element.
    Use config.head_dim when present."""
    raise NotImplementedError("TODO: implement kv_bytes_estimate")


def cache_bytes(cache, num_layers: int) -> int:
    """TODO: sum numel * element_size of keys and values over all layers (use layer_kv)."""
    raise NotImplementedError("TODO: implement cache_bytes")


def measure_request(model, prompt_ids: torch.Tensor, decode_steps: int, use_cache: bool) -> dict:
    """TODO: measure one independent request on CUDA.
      1. gc.collect(); torch.cuda.empty_cache(); baseline = torch.cuda.memory_allocated()
      2. torch.cuda.reset_peak_memory_stats()
      3. cached: prefill (use_cache=True, logits_to_keep=1), then decode_steps one-token steps with
         past_key_values and an attention mask covering past + current tokens.
         uncached: decode_steps + 1 full forwards on the growing sequence (use_cache=False, logits_to_keep=1).
         Choose tokens greedily.
      4. torch.cuda.synchronize(); return, IN MiB:
         {"peak_delta_mib", "retained_delta_mib" (memory_allocated - baseline while the cache is still alive),
          "reserved_mib", "baseline_mib", "cache_mib" (cache_bytes for cached, else 0), "cache_len"}"""
    raise NotImplementedError("TODO: implement measure_request")


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
