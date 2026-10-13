"""Day 2 solution: inspect how text becomes token IDs."""

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
    encoding = tokenizer(text)
    ids = encoding["input_ids"]
    return {
        "input_ids": ids,
        "tokens": tokenizer.convert_ids_to_tokens(ids),
        "num_tokens": len(ids),
        "num_words": len(text.split()),
        "attention_mask": encoding["attention_mask"],
        "decoded": tokenizer.decode(ids, clean_up_tokenization_spaces=False),
    }


def print_encoding(name: str, info: dict) -> None:
    print(f"\n[{name}] tokens={info['num_tokens']} words={info['num_words']}")
    for token_id, piece in zip(info["input_ids"], info["tokens"]):
        print(f"  {token_id:>7}  {piece!r}")


def chat_template_ids(tokenizer, user_text: str) -> list[int]:
    messages = [{"role": "user", "content": user_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(formatted)["input_ids"]


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
