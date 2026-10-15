"""Day 6 exercise: build a causal attention function and validate it against SDPA."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent / "results"
B, H, D = 2, 2, 4
# FP32 tolerances: justify these in your notes (see README).
RTOL, ATOL = 1.3e-6, 1e-5


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """TODO: causal scaled dot-product attention.

    Inputs:  q, k, v shaped [B, H, T, D] with equal T.
    Returns: [B, H, T, D].
    Steps: scores = q @ k^T / sqrt(D) -> add causal -inf mask -> softmax over keys -> @ v.
    Document each intermediate shape in a comment.
    """
    raise NotImplementedError("TODO: implement causal_attention")


def compare_with_sdpa(q, k, v) -> float:
    """TODO: compare causal_attention with
    F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    using torch.testing.assert_close(..., rtol=RTOL, atol=ATOL); return the max abs error."""
    raise NotImplementedError("TODO: implement compare_with_sdpa")


def causal_independence(q, k, v) -> dict:
    """TODO: clone q, k, v; replace only the last position (index -1 on the T axis) with new
    random values; run causal_attention on both versions and return
      {"earlier_max_change": float, "last_max_change": float}.
    Earlier positions must not change; the last position should."""
    raise NotImplementedError("TODO: implement causal_independence")


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
