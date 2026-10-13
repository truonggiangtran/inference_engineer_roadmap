"""Day 10 exercise: RMSNorm and the residual path of a real decoder block."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
HIDDEN, EPS = 8, 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """TODO: x / sqrt(mean(x**2 over the last dim) + eps) * weight."""
    raise NotImplementedError("TODO: implement rms_norm")


def layer_norm_vs_rms_norm(x: torch.Tensor, offset: float) -> dict:
    """TODO: compute LayerNorm (F.layer_norm without weight/bias) and your rms_norm (weight=1)
    on x and on x + offset. Return the max abs change of each:
    {"layer_norm_change": float, "rms_norm_change": float}"""
    raise NotImplementedError("TODO: implement layer_norm_vs_rms_norm")


def trace_block_shapes(model, input_ids: torch.Tensor) -> dict:
    """TODO: register hooks on model.model.layers[0]:
      - forward pre-hook on the block (with_kwargs=True) to capture its input hidden_states
      - forward hooks on input_layernorm, self_attn, post_attention_layernorm, mlp, and the block
    Run model(input_ids) once, remove the hooks, and return a dict of captured tensors with keys
      block_input, input_layernorm, self_attn, post_attention_layernorm, mlp, block_output.
    Notes: self_attn returns a tuple (take [0]); the block may return a tensor or a tuple."""
    raise NotImplementedError("TODO: implement trace_block_shapes")


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, HIDDEN)
    weight = torch.rand(HIDDEN) + 0.5

    ours = rms_norm(x, weight, EPS)
    reference = nn.RMSNorm(HIDDEN, eps=EPS)
    hf_reference = Qwen3RMSNorm(HIDDEN, eps=EPS)
    with torch.no_grad():
        reference.weight.copy_(weight)
        hf_reference.weight.copy_(weight)
        torch.testing.assert_close(ours, reference(x))
        torch.testing.assert_close(ours, hf_reference(x))
    print("rms_norm matches torch.nn.RMSNorm and Qwen3RMSNorm")

    change = layer_norm_vs_rms_norm(x, offset=3.0)
    assert change["layer_norm_change"] < 1e-4 < change["rms_norm_change"]
    print(f"offset +3: LayerNorm change={change['layer_norm_change']:.2e}, RMSNorm change={change['rms_norm_change']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    input_ids = tokenizer("The residual stream carries information.", return_tensors="pt").input_ids.to(DEVICE)

    with torch.inference_mode():
        captured = trace_block_shapes(model, input_ids)
    shapes = {name: list(t.shape) for name, t in captured.items()}
    print(json.dumps(shapes, indent=2))

    h = captured["block_input"] + captured["self_attn"]
    recomputed = h + captured["mlp"]
    torch.testing.assert_close(recomputed, captured["block_output"])
    print("verified: out = (x + attn(norm1(x))) + mlp(norm2(h))")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "block_shapes.json").write_text(json.dumps({"dtype": str(DTYPE), "shapes": shapes}, indent=2))
    print("All Day 10 checks passed.")


if __name__ == "__main__":
    main()
