"""Day 5 solution: scaled dot products, causal masking, and softmax."""

import math

import torch

B, H, T, D = 2, 2, 4, 4


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def attention_weights(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return stable_softmax(scores + mask, dim=-1)


def attention_output(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return weights @ v


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
