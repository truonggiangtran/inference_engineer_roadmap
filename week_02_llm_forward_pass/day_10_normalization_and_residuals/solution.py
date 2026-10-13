"""Day 10 solution: RMSNorm and the residual path of a real decoder block."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
HIDDEN, EPS = 8, 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def layer_norm_vs_rms_norm(x: torch.Tensor, offset: float) -> dict:
    ones = torch.ones(x.shape[-1])
    ln = lambda t: F.layer_norm(t, (t.shape[-1],))
    rms = lambda t: rms_norm(t, ones, EPS)
    return {
        "layer_norm_change": (ln(x + offset) - ln(x)).abs().max().item(),
        "rms_norm_change": (rms(x + offset) - rms(x)).abs().max().item(),
    }


def first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def trace_block_shapes(model, input_ids: torch.Tensor) -> dict:
    block = model.model.layers[0]
    captured = {}

    def capture_input(module, args, kwargs):
        captured["block_input"] = args[0] if args else kwargs["hidden_states"]

    def capture(name):
        def hook(module, args, output):
            captured[name] = first_tensor(output)
        return hook

    handles = [block.register_forward_pre_hook(capture_input, with_kwargs=True)]
    for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
        handles.append(getattr(block, name).register_forward_hook(capture(name)))
    handles.append(block.register_forward_hook(capture("block_output")))
    try:
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


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
