"""Update the captured buffer without replacing its storage."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda


def update_input(case: GraphCase, new: torch.Tensor) -> None:
    # TODO: copy values into persistent storage, not case.x = new.
    raise NotImplementedError("Preserve the address used by the captured graph.")


if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    case.capture()
    pointer = case.x.data_ptr()
    for value in (-2.0, 3.0):
        new = torch.full_like(case.x, value)
        update_input(case, new)
        assert case.x.data_ptr() == pointer
        torch.testing.assert_close(case.x, new)
        expected = case.eager().clone()
        case.replay()
        torch.testing.assert_close(case.output, expected)
