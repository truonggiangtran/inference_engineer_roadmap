"""Day 12 exercise: shifted targets, inference controls, and an uncached greedy loop."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # exact token comparison; BF16 near-ties could flip an argmax
RESULTS_DIR = Path(__file__).parent / "results"
NEW_TOKENS = 12


def shifted_pairs(tokenizer, ids: list[int]) -> list[tuple[str, str]]:
    """TODO: return [(decoded prefix ids[:t+1], decoded target ids[t+1]) for each t]."""
    raise NotImplementedError("TODO: implement shifted_pairs")


def manual_loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    """TODO: logits = model(input_ids).logits; cross-entropy between logits[:, :-1] and
    input_ids[:, 1:] (flatten both). This must equal model(input_ids, labels=input_ids).loss."""
    raise NotImplementedError("TODO: implement manual_loss")


def check_inference_state(model) -> dict:
    """TODO: return {"model_training": model.training,
                     "inference_mode": torch.is_inference_mode_enabled(),
                     "new_tensor_requires_grad": <requires_grad of a tensor computed from a parameter>}"""
    raise NotImplementedError("TODO: implement check_inference_state")


def greedy_uncached(model, input_ids: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """TODO: loop new_tokens times: forward the FULL current ids with use_cache=False, take the
    last-position logits, argmax, append. Return only the new token IDs [1, new_tokens].
    Comment which steps run on CPU vs operate on GPU tensors."""
    raise NotImplementedError("TODO: implement greedy_uncached")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)
    model.eval()

    sample = tokenizer("Tokens are predicted one at a time.")["input_ids"]
    for prefix, target in shifted_pairs(tokenizer, sample):
        print(f"{prefix!r:>45} -> {target!r}")

    with torch.inference_mode():
        state = check_inference_state(model)
        assert state == {"model_training": False, "inference_mode": True, "new_tensor_requires_grad": False}, state

        ids = torch.tensor([sample], device=DEVICE)
        ours = manual_loss(model, ids)
        reference = model(ids, labels=ids).loss
        torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-4)
        print(f"manual shifted loss {ours.item():.4f} == model loss {reference.item():.4f}")

        prompt = tokenizer("The three phases of a language model's life are", return_tensors="pt").to(DEVICE)
        ours_tokens = greedy_uncached(model, prompt["input_ids"], NEW_TOKENS)
        generated = model.generate(
            **prompt, do_sample=False, use_cache=False, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS
        )[:, prompt["input_ids"].shape[1]:]

    assert torch.equal(ours_tokens, generated), f"loop {ours_tokens.tolist()} != generate {generated.tolist()}"
    text = tokenizer.decode(ours_tokens[0])
    print(f"uncached greedy loop matches generate(): {text!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "generation_loop.json").write_text(json.dumps(
        {"inference_state": state, "loss": ours.item(), "tokens": ours_tokens[0].tolist(), "text": text}, indent=2
    ))
    print("All Day 12 checks passed.")


if __name__ == "__main__":
    main()
