"""Day 5 exercise: scaled dot products, causal masking, and softmax."""

import math

import torch

B, H, T, D = 2, 2, 4, 4


def causal_mask(seq_len: int) -> torch.Tensor:
    """TODO: return a float [seq_len, seq_len] additive mask: 0 on and below the diagonal,
    -inf above it. Hint: torch.triu(..., diagonal=1)."""
    raise NotImplementedError("TODO: implement causal_mask")


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """TODO: softmax computed as exp(x - max) / sum(exp(x - max)) along `dim`."""
    raise NotImplementedError("TODO: implement stable_softmax")


def attention_weights(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TODO: [B,H,T,D] x [B,H,T,D] -> [B,H,T,T]: scores / sqrt(D) + mask, softmax over keys."""
    raise NotImplementedError("TODO: implement attention_weights")


def attention_output(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """TODO: [B,H,T,T] @ [B,H,T,D] -> [B,H,T,D]."""
    raise NotImplementedError("TODO: implement attention_output")


def main() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))

    mask = causal_mask(T)
    assert mask.shape == (T, T)
    assert torch.all(mask.tril() == 0) and torch.isinf(mask.triu(1)[mask.triu(1) != 0]).all()

    big = torch.tensor([[1000.0, 999.0, 998.0], [1.0, 2.0, 3.0]])
    stable = stable_softmax(big)
    assert torch.isfinite(stable).all(), "stable softmax must not overflow"
    torch.testing.assert_close(stable, torch.softmax(big, dim=-1))

    weights = attention_weights(q, k, mask)
    assert weights.shape == (B, H, T, T)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B, H, T))
    assert torch.all(weights.triu(1) == 0), "future positions must receive exactly zero weight"

    out = attention_output(weights, v)
    assert out.shape == (B, H, T, D)
    b, h, i = 0, 1, 2
    manual = sum(weights[b, h, i, j] * v[b, h, j] for j in range(T))
    torch.testing.assert_close(out[b, h, i], manual)

    print(f"query row {i} (batch {b}, head {h}) weights: {weights[b, h, i].tolist()}")
    print(f"it attends to tokens 0..{i}; tokens {i + 1}..{T - 1} get zero weight")
    print("All Day 5 checks passed.")


if __name__ == "__main__":
    main()
