"""Day 31: mutate persistent input storage and compare changed replay outputs."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda, save

if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    case.capture()
    pointer = case.x.data_ptr()
    errors = []
    for value in (-2.0, 0.0, 3.0):
        case.x.copy_(torch.full_like(case.x, value))
        expected = case.eager().clone()
        case.replay()
        torch.testing.assert_close(case.output, expected)
        assert case.x.data_ptr() == pointer
        errors.append((case.output - expected).abs().max().item())
    save(Path(__file__).parent / "results", "correctness.json",
         {"max_abs_errors": errors, "persistent_input_pointer": pointer})
