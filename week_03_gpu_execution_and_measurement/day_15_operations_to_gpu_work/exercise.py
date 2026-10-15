"""Day 15 exercise: map model operators to GPU work and separate timing scopes."""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
PROMPT = "Inference engineers map every model operation to the GPU work it launches. " * 8


def classify_op(name: str) -> str:
    """TODO: return one of "matmul", "attention", "normalization", "elementwise", "copy",
    "indexing", "other" for an aten op name such as "aten::addmm" or "aten::rsqrt".
    Check attention before matmul (attention ops may contain "mm" in their internals)."""
    raise NotImplementedError("TODO: implement classify_op")


def profile_forward(model, input_ids: torch.Tensor):
    """TODO: run 3 warmup forwards, then profile one forward with
    profile(activities=[CPU] + ([CUDA] if DEVICE == "cuda" else [])). Return the profiler."""
    raise NotImplementedError("TODO: implement profile_forward")


def summarize(prof) -> dict:
    """TODO: return {
        "cpu_ms_by_category": {category: total self CPU ms of aten:: ops},
        "aten_calls": number of aten:: operator calls,
        "gpu_kernels": number of CUDA kernel events (0 on CPU),
        "top_ops": [(key, count, category) for the 10 aten ops with highest self CPU time]}
    Hints: prof.key_averages() entries have .key, .count, .self_cpu_time_total (microseconds);
    for kernels use [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]."""
    raise NotImplementedError("TODO: implement summarize")


def end_to_end_breakdown(model, tokenizer, prompt: str) -> dict:
    """TODO: time each scope in milliseconds:
      tokenize_ms (perf_counter), h2d_ms (copy to DEVICE + synchronize),
      forward_ms (CUDA events around the forward; synchronized perf_counter on CPU),
      select_and_decode_ms (argmax of last logits + .item() + tokenizer.decode), total_ms.
    Return the dict."""
    raise NotImplementedError("TODO: implement end_to_end_breakdown")


def main() -> None:
    for name, expected in [("aten::addmm", "matmul"), ("aten::scaled_dot_product_attention", "attention"),
                           ("aten::rsqrt", "normalization"), ("aten::silu", "elementwise"),
                           ("aten::embedding", "indexing"), ("aten::copy_", "copy")]:
        assert classify_op(name) == expected, f"{name} -> {classify_op(name)}, expected {expected}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    with torch.inference_mode():
        prof = profile_forward(model, input_ids)
        summary = summarize(prof)
        breakdown = [end_to_end_breakdown(model, tokenizer, PROMPT) for _ in range(5)]

    summary["prompt_tokens"] = input_ids.shape[1]
    summary["device"], summary["dtype"] = DEVICE, str(DTYPE)
    summary["end_to_end_runs_ms"] = breakdown
    print(json.dumps(summary, indent=2))
    if DEVICE == "cuda":
        print(f"{summary['aten_calls']} aten calls launched {summary['gpu_kernels']} GPU kernels")
    else:
        print("CPU only: no GPU kernels; these are not GPU measurements.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "op_map.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
