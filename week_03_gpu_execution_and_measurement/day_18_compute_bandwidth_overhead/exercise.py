"""Day 18 exercise: arithmetic intensity, achieved throughput, and launch overhead."""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
M_VALUES = (1, 16, 128, 512)
WARMUP, REPEATS = 10, 50
# Fill in from NVIDIA's RTX 5070 specifications and cite the source in your notes.
PEAK_TFLOPS = None  # dense BF16 tensor-core TFLOP/s
PEAK_GBPS = None  # DRAM bandwidth in GB/s


def linear_flops(m: int, k: int, n: int) -> int:
    """TODO: multiply-adds count as 2 FLOPs."""
    raise NotImplementedError("TODO: implement linear_flops")


def linear_min_bytes(m: int, k: int, n: int, bytes_per_element: int) -> int:
    """TODO: read input [m,k] + weight [n,k] once, write output [m,n] once."""
    raise NotImplementedError("TODO: implement linear_min_bytes")


def arithmetic_intensity(m: int, k: int, n: int, bytes_per_element: int) -> float:
    """TODO: FLOPs / min bytes."""
    raise NotImplementedError("TODO: implement arithmetic_intensity")


def median_ms(fn) -> float:
    """Median time of fn(): CUDA events on GPU, synchronized perf_counter on CPU."""
    for _ in range(WARMUP):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        pairs = []
        for _ in range(REPEATS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) for s, e in pairs)
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def measure_linear(m: int, k: int, n: int) -> dict:
    """TODO: preallocate x [m,k] and w [n,k] on DEVICE in DTYPE, time F.linear(x, w) with median_ms,
    and return {"m","k","n","ms","ai","achieved_tflops","achieved_gbps"}."""
    raise NotImplementedError("TODO: implement measure_linear")


def launch_overhead(launches: int = 1000) -> dict:
    """TODO: x = 1-element tensor. Return
      {"gpu_ms_per_kernel": median_ms(lambda: x.add_(1)),
       "host_us_per_launch": host time of `launches` x.add_(1) calls WITHOUT sync, divided by launches}.
    Synchronize after the host-timed loop."""
    raise NotImplementedError("TODO: implement launch_overhead")


def main() -> None:
    assert linear_flops(2, 3, 4) == 48
    assert linear_min_bytes(2, 3, 4, 2) == (6 + 12 + 8) * 2

    config = AutoConfig.from_pretrained(MODEL_ID)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    shapes = {
        "q_proj": (config.hidden_size, config.num_attention_heads * head_dim),
        "gate_proj": (config.hidden_size, config.intermediate_size),
    }
    rows = []
    with torch.inference_mode():
        for name, (k, n) in shapes.items():
            for m in M_VALUES:
                row = {"layer": name, **measure_linear(m, k, n)}
                rows.append(row)
                print(f"{name:<9} M={m:<4} {row['ms']:.4f} ms  AI={row['ai']:7.1f} FLOP/B  "
                      f"{row['achieved_tflops']:7.2f} TFLOP/s  {row['achieved_gbps']:8.1f} GB/s")
        overhead = launch_overhead()
    print(f"launch overhead: {overhead}")

    report = {"device": DEVICE, "dtype": str(DTYPE), "linear": rows, "launch_overhead": overhead,
              "peak_tflops": PEAK_TFLOPS, "peak_gbps": PEAK_GBPS}
    if PEAK_TFLOPS and PEAK_GBPS:
        ridge = PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9)
        report["ridge_flops_per_byte"] = ridge
        for row in rows:
            row["side_of_ridge"] = "bandwidth-bound region" if row["ai"] < ridge else "compute-bound region"
        print(f"ridge point: {ridge:.1f} FLOP/byte")
    else:
        print("Set PEAK_TFLOPS and PEAK_GBPS to compute the ridge point.")
    if DEVICE != "cuda":
        print("CPU only: these are CPU timings, not GPU measurements.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "bottleneck.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
