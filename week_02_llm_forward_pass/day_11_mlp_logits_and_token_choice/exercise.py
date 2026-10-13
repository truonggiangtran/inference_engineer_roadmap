"""Day 11 exercise: gated MLP, logits, probabilities, and token choice."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
HIDDEN, INTERMEDIATE = 8, 16


def gated_mlp(x, gate_w, up_w, down_w) -> torch.Tensor:
    """TODO: down(silu(gate(x)) * up(x)). Weights use nn.Linear layout [out, in].
    Annotate the shape after each step."""
    raise NotImplementedError("TODO: implement gated_mlp")


def probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """TODO: softmax(logits / temperature) over the last dim."""
    raise NotImplementedError("TODO: implement probabilities")


def entropy(probs: torch.Tensor) -> float:
    """TODO: -sum(p * log(p)), ignoring zero probabilities."""
    raise NotImplementedError("TODO: implement entropy")


def last_position_logits(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: return (logits_from_model, logits_from_lm_head), both [vocab] for the last position.
    logits_from_model: model(input_ids).logits[0, -1]
    logits_from_lm_head: model.lm_head(model.model(input_ids).last_hidden_state)[0, -1]
    (model.model's last_hidden_state already includes the final RMSNorm.)"""
    raise NotImplementedError("TODO: implement last_position_logits")


def main() -> None:
    torch.manual_seed(0)
    config = Qwen3Config(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, hidden_act="silu")
    reference = Qwen3MLP(config)
    x = torch.randn(2, 3, HIDDEN)
    with torch.no_grad():
        ours = gated_mlp(x, reference.gate_proj.weight, reference.up_proj.weight, reference.down_proj.weight)
        torch.testing.assert_close(ours, reference(x))
    assert ours.shape == x.shape
    print(f"gated MLP matches Qwen3MLP: {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}")

    tiny = torch.tensor([2.0, 1.0, 0.5, -1.0])
    rows = []
    for tau in (0.5, 1.0, 2.0):
        p = probabilities(tiny, tau)
        torch.testing.assert_close(p.sum(), torch.tensor(1.0))
        assert p.argmax() == tiny.argmax(), "temperature must not change the argmax"
        rows.append({"temperature": tau, "probs": [round(v, 4) for v in p.tolist()], "entropy": entropy(p)})
    assert rows[0]["entropy"] < rows[1]["entropy"] < rows[2]["entropy"]
    for row in rows:
        print(row)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode():
        from_model, from_head = last_position_logits(model, input_ids)
    torch.testing.assert_close(from_model, from_head)
    assert from_model.shape == (model.config.vocab_size,)

    probs = torch.softmax(from_model.float(), dim=-1)
    top = torch.topk(probs, 5)
    top5 = [{"token": tokenizer.decode([i]), "id": i, "prob": round(p, 4)} for p, i in zip(top.values.tolist(), top.indices.tolist())]
    print("top-5 next tokens:", top5)
    print(f"greedy choice: {tokenizer.decode([from_model.argmax().item()])!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "logits.json").write_text(json.dumps({"temperature": rows, "top5": top5}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("All Day 11 checks passed.")


if __name__ == "__main__":
    main()
