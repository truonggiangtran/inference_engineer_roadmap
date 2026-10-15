"""Day 34: diagnostic trace only, never used as benchmark timing."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda

if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    case.capture()
    case.check()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        with torch.profiler.record_function("eager_submission"):
            for _ in range(10):
                case.eager()
        torch.cuda.synchronize()
        with torch.profiler.record_function("graph_submission"):
            for _ in range(10):
                case.replay()
        torch.cuda.synchronize()
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    profile.export_chrome_trace(str(results / "graph.trace.json"))
    print("Diagnostic trace written; compare against unprofiled Day 32 timings.")
