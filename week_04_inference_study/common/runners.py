"""Teacher-forced cached and uncached request runners with phase timing.

Both modes forward the same prompt and the same recorded continuation, request only
last-position logits (logits_to_keep=1), and pass an all-ones attention mask covering
past and current tokens. They differ only in whether a KV cache is reused.
"""

import time
from dataclasses import dataclass, field

import torch


def layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keys, values) for one layer across Hugging Face cache layouts."""
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def cache_bytes(cache, num_layers: int) -> int:
    total = 0
    for layer in range(num_layers):
        keys, values = layer_kv(cache, layer)
        total += keys.numel() * keys.element_size() + values.numel() * values.element_size()
    return total


def kv_bytes_estimate(config, batch: int, tokens: int, bytes_per_element: int) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * batch * tokens * config.num_key_value_heads * head_dim * bytes_per_element


class PhaseTimer:
    """Marks phase boundaries. On CUDA, intervals are stream-timeline time between events,
    which includes GPU idle gaps caused by host (Python/launch) overhead."""

    def __init__(self, device: torch.device):
        self.cuda = torch.device(device).type == "cuda"
        self.marks = {}

    def mark(self, name: str) -> None:
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks[name] = event
        else:
            self.marks[name] = time.perf_counter()

    def elapsed_ms(self, start: str, end: str) -> float:
        if self.cuda:
            self.marks[end].synchronize()
            return self.marks[start].elapsed_time(self.marks[end])
        return (self.marks[end] - self.marks[start]) * 1000


@dataclass
class RequestResult:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    logits: list[torch.Tensor] = field(default_factory=list)  # float32 CPU, prefill + each decode step
    cache_lengths: list[int] = field(default_factory=list)
    cache: object | None = None


def run_request(model, prompt_ids: torch.Tensor, continuation_ids: torch.Tensor, mode: str, *,
                timer: PhaseTimer | None = None, keep_logits: bool = False, keep_cache: bool = False) -> RequestResult:
    """Run one independent teacher-forced request: prefill, then one forward per continuation token.

    keep_logits copies logits to CPU at every step (a sync + transfer): use it only for
    correctness runs, never for timing runs.
    """
    if mode not in ("cached", "uncached"):
        raise ValueError(f"mode must be 'cached' or 'uncached', got {mode!r}")
    prompt_len, steps = prompt_ids.shape[1], continuation_ids.shape[1]
    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    full_mask = torch.ones_like(full_ids)
    result = RequestResult()
    cache = None

    if timer:
        timer.mark("start")
    if mode == "cached":
        out = model(input_ids=prompt_ids, attention_mask=full_mask[:, :prompt_len], use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        result.cache_lengths.append(cache.get_seq_length())
    else:
        out = model(input_ids=prompt_ids, attention_mask=full_mask[:, :prompt_len], use_cache=False, logits_to_keep=1)
    if keep_logits:
        result.logits.append(out.logits[0, -1].float().cpu())
    if timer:
        timer.mark("prefill_end")

    for i in range(steps):
        length = prompt_len + i + 1
        if mode == "cached":
            out = model(input_ids=continuation_ids[:, i:i + 1], attention_mask=full_mask[:, :length],
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            result.cache_lengths.append(cache.get_seq_length())
        else:
            out = model(input_ids=full_ids[:, :length], attention_mask=full_mask[:, :length],
                        use_cache=False, logits_to_keep=1)
        if keep_logits:
            result.logits.append(out.logits[0, -1].float().cpu())
    if timer:
        timer.mark("decode_end")
        result.prefill_ms = timer.elapsed_ms("start", "prefill_end")
        result.decode_ms = timer.elapsed_ms("prefill_end", "decode_end")
    if keep_cache:
        result.cache = cache
    return result
