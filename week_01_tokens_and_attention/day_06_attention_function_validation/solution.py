"""Day 6 solution: build a causal attention function and validate it against SDPA."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent / "results"
B, H, D = 2, 2, 4
# FP32 defaults of assert_close; short reductions in FP32 stay well inside these.
RTOL, ATOL = 1.3e-6, 1e-5


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention.

    Inputs:  q, k, v shaped [B, H, T, D] with equal T.
    Returns: [B, H, T, D].
    """
    seq_len, head_dim = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # [B, H, T, T]
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1
    )  # [T, T], broadcast over B and H
    weights = torch.softmax(scores + mask, dim=-1)  # [B, H, T, T], rows sum to 1
    return weights @ v  # [B, H, T, D]


def compare_with_sdpa(q, k, v) -> float:
    actual = causal_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return (actual - expected).abs().max().item()


def causal_independence(q, k, v) -> dict:
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for t in (q2, k2, v2):
        t[..., -1, :] = torch.randn_like(t[..., -1, :])
    before = causal_attention(q, k, v)
    after = causal_attention(q2, k2, v2)
    return {
        "earlier_max_change": (after[..., :-1, :] - before[..., :-1, :]).abs().max().item(),
        "last_max_change": (after[..., -1, :] - before[..., -1, :]).abs().max().item(),
    }


def main() -> None:
    report = {"dtype": "float32", "rtol": RTOL, "atol": ATOL, "runs": []}
    for seq_len in (4, 7):
        torch.manual_seed(seq_len)
        q, k, v = (torch.randn(B, H, seq_len, D) for _ in range(3))

        out = causal_attention(q, k, v)
        assert out.shape == (B, H, seq_len, D)
        max_error = compare_with_sdpa(q, k, v)
        independence = causal_independence(q, k, v)
        assert independence["earlier_max_change"] <= ATOL, "earlier outputs changed: mask is wrong"
        assert independence["last_max_change"] > 1e-3, "perturbation had no effect: check the test"

        run = {
            "seq_len": seq_len,
            "shapes": {
                "q/k/v": [B, H, seq_len, D],
                "scores/weights": [B, H, seq_len, seq_len],
                "output": list(out.shape),
            },
            "max_abs_error_vs_sdpa": max_error,
            **independence,
        }
        report["runs"].append(run)
        print(json.dumps(run, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "attention_validation.json").write_text(json.dumps(report, indent=2))
    print(f"All Day 6 checks passed. Saved {RESULTS_DIR / 'attention_validation.json'}")


if __name__ == "__main__":
    main()
