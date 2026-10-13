"""Day 16 solution: compare GPU timing methods, warmup, and statistics."""

import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils import benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SIZE = 4096 if DEVICE == "cuda" else 1024
WARMUP, REPEATS = 5, 30
RESULTS_DIR = Path(__file__).parent / "results"


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_cold(fn) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1000


def time_unsynchronized(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    sync()
    return samples


def time_synchronized(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def time_cuda_events(fn, repeats: int) -> list[float]:
    if DEVICE != "cuda":
        return time_synchronized(fn, repeats)
    pairs = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def summarize(samples: list[float]) -> dict:
    deciles = statistics.quantiles(samples, n=10)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": deciles[0],
        "p90_ms": deciles[-1],
        "stdev_ms": statistics.stdev(samples),
        "min_ms": min(samples),
        "n": len(samples),
    }


def main() -> None:
    a = torch.randn(SIZE, SIZE, device=DEVICE, dtype=DTYPE)  # preallocated outside timing
    b = torch.randn(SIZE, SIZE, device=DEVICE, dtype=DTYPE)
    fn = lambda: a @ b

    cold = time_cold(fn)
    for _ in range(WARMUP):
        fn()
    sync()

    results = {
        "device": DEVICE, "dtype": str(DTYPE), "size": SIZE, "warmup": WARMUP,
        "cold_first_call_ms": cold,
        "unsynchronized": summarize(time_unsynchronized(fn, REPEATS)),
        "synchronized": summarize(time_synchronized(fn, REPEATS)),
        "cuda_events": summarize(time_cuda_events(fn, REPEATS)),
    }
    timer = benchmark.Timer(stmt="a @ b", globals={"a": a, "b": b})
    results["torch_benchmark_median_ms"] = timer.blocked_autorange(min_run_time=1.0).median * 1000

    events_median = results["cuda_events"]["median_ms"]
    results["matmul_tflops_at_median"] = 2 * SIZE**3 / (events_median / 1000) / 1e12
    print(json.dumps(results, indent=2))
    if DEVICE == "cuda":
        ratio = results["unsynchronized"]["median_ms"] / events_median
        print(f"unsynchronized / events median = {ratio:.3f}: the host timer measured submission, not execution")
    else:
        print("CPU only: these are CPU timings, not GPU measurements.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "timing_methods.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
