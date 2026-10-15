"""Small fixed-shape CUDA workload and explicit benchmark boundaries."""

import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


def require_cuda() -> bool:
    if not torch.cuda.is_available():
        print("SKIP: CUDA Graph experiments require CUDA. No CPU timing is substituted.")
        return False
    return True


class GraphCase:
    def __init__(self, size: int, steps: int = 8):
        self.x = torch.randn(size, device="cuda")
        self.temp = torch.empty_like(self.x)
        self.output = torch.empty_like(self.x)
        self.steps = steps
        self.graph: torch.cuda.CUDAGraph | None = None

    @torch.no_grad()
    def eager(self) -> torch.Tensor:
        source = self.x
        for _ in range(self.steps):
            torch.mul(source, 1.0001, out=self.temp)
            torch.add(self.temp, 0.01, out=self.output)
            source = self.output
        return self.output

    def capture(self) -> float:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.eager()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.eager()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    def replay(self) -> None:
        if self.graph is None:
            raise RuntimeError("Capture before replay.")
        self.graph.replay()

    def check(self) -> None:
        for value in (-2.0, 0.0, 3.0):
            self.x.fill_(value)
            expected = self.eager().clone()
            self.replay()
            torch.testing.assert_close(self.output, expected)
        torch.cuda.synchronize()


def measure(fn: Callable[[], object], repeats: int = 30) -> dict:
    if repeats < 2:
        raise ValueError("At least two repeats are required.")
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        gpu.append(begin.elapsed_time(end))
    return {
        "n": repeats,
        "wall_ms": wall,
        "gpu_ms": gpu,
        "wall_median_ms": statistics.median(wall),
        "gpu_median_ms": statistics.median(gpu),
    }


def compare(size: int) -> dict:
    case = GraphCase(size)
    eager = measure(case.eager)
    before = torch.cuda.memory_allocated()
    setup_ms = case.capture()
    capture_allocated_delta = torch.cuda.memory_allocated() - before
    case.check()
    replay = measure(case.replay)
    return {
        "size": size, "dtype": str(case.x.dtype), "steps": case.steps,
        "capture_ms_excluding_warmup": setup_ms,
        "capture_allocated_delta_bytes": capture_allocated_delta,
        "eager": eager, "replay": replay,
        "correctness": "assert_close passed for three changed inputs",
        "boundary": "preallocated input; no host/device copies; synchronized wall time",
        "order": "eager then replay; repeat in reverse order before strong claims",
    }


def save(directory: Path, name: str, data: object) -> None:
    directory.mkdir(exist_ok=True)
    payload = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "data": data,
    }
    (directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
