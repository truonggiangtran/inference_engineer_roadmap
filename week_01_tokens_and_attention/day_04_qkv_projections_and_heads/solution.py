"""Day 4 solution: Q/K/V projections and the multi-head layout."""

import torch
from torch import nn

B, T, HIDDEN, H, D = 2, 4, 8, 2, 4


def make_projections(seed: int = 0) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    torch.manual_seed(seed)
    return tuple(nn.Linear(HIDDEN, H * D, bias=False) for _ in range(3))


def project_qkv(x: torch.Tensor, projections) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_proj, k_proj, v_proj = projections
    # q == x @ q_proj.weight.T: a GEMM with M = B*T, K = HIDDEN, N = H*D.
    return q_proj(x), k_proj(x), v_proj(x)


def split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq, width = t.shape
    return t.view(batch, seq, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(t: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, head_dim = t.shape
    return t.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)


def describe_layout(name: str, t: torch.Tensor) -> None:
    print(f"{name:<22} shape={str(tuple(t.shape)):<14} strides={str(t.stride()):<16} "
          f"contiguous={t.is_contiguous()!s:<5} data_ptr={t.data_ptr()}")


def main() -> None:
    torch.manual_seed(1)
    x = torch.randn(B, T, HIDDEN)
    projections = make_projections()
    with torch.no_grad():
        q, k, v = project_qkv(x, projections)
    for t in (q, k, v):
        assert t.shape == (B, T, H * D)
    torch.testing.assert_close(q, x @ projections[0].weight.T)

    q_heads = split_heads(q, H)
    assert q_heads.shape == (B, H, T, D), f"expected {(B, H, T, D)}, got {tuple(q_heads.shape)}"
    for h in range(H):
        torch.testing.assert_close(q_heads[:, h], q[:, :, h * D:(h + 1) * D])
    assert q_heads.data_ptr() == q.data_ptr(), "split_heads should return a view, not a copy"

    describe_layout("q [B,T,H*D]", q)
    describe_layout("q.view [B,T,H,D]", q.view(B, T, H, D))
    describe_layout("q_heads [B,H,T,D]", q_heads)
    describe_layout("q_heads.contiguous()", q_heads.contiguous())

    try:
        q_heads.view(-1)
        raise AssertionError("expected view() on a non-contiguous tensor to fail")
    except RuntimeError:
        print("q_heads.view(-1) fails as expected; flattening needs .contiguous() or .reshape()")
    assert q_heads.contiguous().view(-1).shape == (B * H * T * D,)

    torch.testing.assert_close(merge_heads(q_heads), q)
    print("All Day 4 checks passed.")


if __name__ == "__main__":
    main()
