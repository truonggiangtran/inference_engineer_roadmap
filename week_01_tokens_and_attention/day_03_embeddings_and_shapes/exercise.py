"""Day 3 exercise: embedding lookup, shapes, indexing, and devices."""

import torch
import torch.nn.functional as F
from torch import nn

VOCAB, HIDDEN = 16, 8


def make_embedding(seed: int = 0) -> nn.Embedding:
    """TODO: seed torch (torch.manual_seed) and return nn.Embedding(VOCAB, HIDDEN)."""
    raise NotImplementedError("TODO: implement make_embedding")


def lookup(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """TODO: return the embedding vectors for ids. Expected shape: [*ids.shape, HIDDEN]."""
    raise NotImplementedError("TODO: implement lookup")


def one_hot_matmul(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """TODO: compute the same lookup as one_hot(ids, VOCAB).float() @ emb.weight.
    Annotate the intermediate one-hot shape in a comment."""
    raise NotImplementedError("TODO: implement one_hot_matmul")


def lookup_on_device(emb: nn.Embedding, ids: torch.Tensor, device: str) -> torch.Tensor:
    """TODO: move both the module and the ids to `device`, look up, and return the result on CPU.
    Do not modify the original module (hint: copy.deepcopy or build a new module)."""
    raise NotImplementedError("TODO: implement lookup_on_device")


def main() -> None:
    emb = make_embedding()
    ids = torch.tensor([[1, 5, 5, 9], [0, 15, 3, 7]])
    print(f"ids: shape={tuple(ids.shape)} dtype={ids.dtype}")
    print(f"weight: shape={tuple(emb.weight.shape)}")

    out = lookup(emb, ids)
    assert out.shape == (2, 4, 8), f"expected (2, 4, 8), got {tuple(out.shape)}"
    torch.testing.assert_close(out, emb.weight[ids])
    torch.testing.assert_close(out[0, 1], out[0, 2], msg="same token ID must give the same vector")
    print(f"output: shape={tuple(out.shape)} = [batch, seq, hidden]")

    via_matmul = one_hot_matmul(emb, ids)
    torch.testing.assert_close(via_matmul, out)
    print("one-hot matmul matches indexing (but reads the whole table for every token)")

    print(f"one token   out[0, 2]  shape={tuple(out[0, 2].shape)}")
    print(f"one sequence out[1]    shape={tuple(out[1].shape)}")
    print(f"last tokens out[:, -1] shape={tuple(out[:, -1].shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_out = lookup_on_device(emb, ids, device)
    torch.testing.assert_close(device_out, out.detach())
    print(f"lookup on {device} matches CPU" + ("" if device == "cuda" else " (CUDA unavailable: CPU only)"))
    print("All Day 3 checks passed.")


if __name__ == "__main__":
    main()
