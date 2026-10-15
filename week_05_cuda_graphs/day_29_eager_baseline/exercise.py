"""Implement a synchronized end-to-end timer; compare with common.measure."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda


def elapsed_ms(fn) -> float:
    # TODO: synchronize before starting and after submitting the work.
    raise NotImplementedError("Implement wall timing with explicit GPU completion.")


if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    for _ in range(5):
        case.eager()
    samples = [elapsed_ms(case.eager) for _ in range(30)]
    assert all(value > 0 for value in samples)
    print(samples)
