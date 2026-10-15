"""Warm up on a side stream, then capture case.eager into a CUDAGraph."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda


def capture(case: GraphCase) -> torch.cuda.CUDAGraph:
    # TODO: order side-stream warmup correctly, retain the graph, and capture.
    raise NotImplementedError("Implement capture; inspect GraphCase.capture afterward.")


if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    expected = case.eager().clone()
    graph = capture(case)
    graph.replay()
    torch.testing.assert_close(case.output, expected)
