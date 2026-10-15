"""Day 2 exercise: inspect how text becomes token IDs."""

import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).parent / "results"

SAMPLES = {
    "sentence": "Inference engineers measure latency carefully.",
    "code": "def add(a, b):\n    return a + b",
    "punctuation_spaces": "Wait...   what?!  Tokens != words;   spaces   matter.",
}


def describe_encoding(tokenizer, text: str) -> dict:
    """TODO: return a dict with keys:
      input_ids (list[int]), tokens (list[str]), num_tokens (int), num_words (int),
      attention_mask (list[int]), decoded (str)
    Hints: tokenizer(text) returns a dict-like object; tokenizer.convert_ids_to_tokens(ids);
    tokenizer.decode(ids, clean_up_tokenization_spaces=False); count words with text.split().
    """
    raise NotImplementedError("TODO: implement describe_encoding")


def print_encoding(name: str, info: dict) -> None:
    """TODO: print the name, token/word counts, and each (id, token piece) pair."""
    raise NotImplementedError("TODO: implement print_encoding")


def chat_template_ids(tokenizer, user_text: str) -> list[int]:
    """TODO: build messages = [{"role": "user", "content": user_text}] and return the IDs of
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False) after tokenizing that formatted string."""
    raise NotImplementedError("TODO: implement chat_template_ids")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"vocabulary size: {len(tokenizer)}")

    results = {}
    for name, text in SAMPLES.items():
        info = describe_encoding(tokenizer, text)
        assert info["num_tokens"] == len(info["input_ids"]) == len(info["tokens"])
        assert all(m == 1 for m in info["attention_mask"]), "no padding expected for a single string"
        assert info["decoded"] == text, "decode(encode(text)) should round-trip for byte-level BPE"
        print_encoding(name, info)
        results[name] = info

    raw = describe_encoding(tokenizer, SAMPLES["sentence"])
    templated = chat_template_ids(tokenizer, SAMPLES["sentence"])
    assert len(templated) > raw["num_tokens"], "the chat template should add role/control tokens"
    print(f"\nraw tokens: {raw['num_tokens']}, chat-template tokens: {len(templated)}")
    print(tokenizer.decode(templated))
    results["chat_template"] = {"ids": templated, "text": tokenizer.decode(templated)}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tokenization.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {RESULTS_DIR / 'tokenization.json'}")


if __name__ == "__main__":
    main()
