"""Day 15 solution: map model operators to GPU work and separate timing scopes."""

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

CATEGORIES = [
    ("attention", ("scaled_dot_product", "flash_attention", "efficient_attention", "cudnn_attention", "softmax")),
    ("matmul", ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear", "aten::baddbmm")),
    ("normalization", ("rsqrt", "aten::mean", "aten::pow", "rms_norm", "layer_norm")),
    ("indexing", ("embedding", "index_select", "aten::index", "gather", "aten::slice", "aten::select")),
    ("copy", ("copy_", "aten::cat", "contiguous", "clone", "aten::to", "_to_copy", "expand", "reshape", "view", "transpose")),
    ("elementwise", ("aten::add", "aten::mul", "aten::silu", "aten::neg", "aten::sigmoid", "aten::div", "aten::sub", "aten::cos", "aten::sin")),
]


def classify_op(name: str) -> str:
    for category, patterns in CATEGORIES:
        if any(pattern in name for pattern in patterns):
            return category
    return "other"


def profile_forward(model, input_ids: torch.Tensor):
    for _ in range(3):
        model(input_ids)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    with profile(activities=activities) as prof:
        model(input_ids)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    return prof


def summarize(prof) -> dict:
    by_category = defaultdict(float)
    aten_calls = 0
    aten_ops = []
    for evt in prof.key_averages():
        if not evt.key.startswith("aten::"):
            continue
        category = classify_op(evt.key)
        by_category[category] += evt.self_cpu_time_total / 1000
        aten_calls += evt.count
        aten_ops.append((evt.self_cpu_time_total, evt.key, evt.count, category))
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    top = sorted(aten_ops, reverse=True)[:10]
    return {
        "cpu_ms_by_category": {k: round(v, 3) for k, v in sorted(by_category.items(), key=lambda kv: -kv[1])},
        "aten_calls": aten_calls,
        "gpu_kernels": len(kernels),
        "top_ops": [(key, count, category) for _, key, count, category in top],
    }


def end_to_end_breakdown(model, tokenizer, prompt: str) -> dict:
    t0 = time.perf_counter()
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    t1 = time.perf_counter()
    ids = ids.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    if DEVICE == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(ids, logits_to_keep=1).logits
        end.record()
        torch.cuda.synchronize()
        forward_ms = start.elapsed_time(end)
    else:
        f0 = time.perf_counter()
        logits = model(ids, logits_to_keep=1).logits
        forward_ms = (time.perf_counter() - f0) * 1000
    t3 = time.perf_counter()

    token = tokenizer.decode([logits[0, -1].argmax().item()])
    t4 = time.perf_counter()
    return {
        "tokenize_ms": (t1 - t0) * 1000,
        "h2d_ms": (t2 - t1) * 1000,
        "forward_ms": forward_ms,
        "select_and_decode_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
        "token": token,
    }


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
