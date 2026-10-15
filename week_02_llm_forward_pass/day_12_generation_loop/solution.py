"""Day 12 solution: shifted targets, inference controls, and an uncached greedy loop."""

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
    return [(tokenizer.decode(ids[: t + 1]), tokenizer.decode([ids[t + 1]])) for t in range(len(ids) - 1)]


def manual_loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits  # [1, T, vocab]
    predictions = logits[:, :-1]  # positions 0..T-2 predict...
    targets = input_ids[:, 1:]  # ...tokens 1..T-1
    return F.cross_entropy(predictions.reshape(-1, predictions.shape[-1]).float(), targets.reshape(-1))


def check_inference_state(model) -> dict:
    derived = next(model.parameters()) * 1
    return {
        "model_training": model.training,
        "inference_mode": torch.is_inference_mode_enabled(),
        "new_tensor_requires_grad": derived.requires_grad,
    }


def greedy_uncached(model, input_ids: torch.Tensor, new_tokens: int) -> torch.Tensor:
    ids = input_ids
    for _ in range(new_tokens):  # CPU: Python loop and kernel launches
        logits = model(input_ids=ids, use_cache=False).logits  # GPU: full forward over every token
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)  # GPU: argmax over the vocabulary
        ids = torch.cat([ids, next_id], dim=1)  # GPU: new, longer tensor for the next step
    return ids[:, input_ids.shape[1]:]


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
