"""Day 22 solution: build exact-length workloads and run the uncached baseline."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DECODE_STEPS, MANIFEST_PATH, MODEL_ID, PROMPT_LENGTHS, environment_info, load_model, run_request

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
SEED_TEXT = (
    "Inference engineering is the practice of making trained models answer quickly, correctly, "
    "and affordably. It combines model architecture, numerical precision, memory traffic, and "
    "careful measurement. "
)


def build_exact_length_prompt(tokenizer, seed_text: str, length: int) -> list[int]:
    seed_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    repeats = -(-length // len(seed_ids))
    return (seed_ids * repeats)[:length]


def record_continuation(model, prompt_ids: list[int], steps: int) -> list[int]:
    prompt = torch.tensor([prompt_ids], device=model.device)
    output = model.generate(
        input_ids=prompt, attention_mask=torch.ones_like(prompt),
        do_sample=False, min_new_tokens=steps, max_new_tokens=steps,
    )
    return output[0, len(prompt_ids):].tolist()


def main() -> None:
    model, tokenizer = load_model(DTYPE, DEVICE)
    manifest = {
        "model_id": MODEL_ID,
        "revision": getattr(model.config, "_commit_hash", None),
        "benchmark_type": "teacher-forced inference benchmark (not live text generation)",
        "prompt_method": "seed text tokenized without special tokens, repeated, truncated to exact length",
        "continuation_method": f"greedy generate, min_new_tokens=max_new_tokens={DECODE_STEPS}, dtype={DTYPE}",
        "seed_text": SEED_TEXT,
        "decode_steps": DECODE_STEPS,
        "environment": environment_info(model),
        "workloads": {},
    }

    with torch.inference_mode():
        for length in PROMPT_LENGTHS:
            prompt = build_exact_length_prompt(tokenizer, SEED_TEXT, length)
            assert len(prompt) == length, f"prompt has {len(prompt)} tokens, expected {length}"
            continuation = record_continuation(model, prompt, DECODE_STEPS)
            assert len(continuation) == DECODE_STEPS
            manifest["workloads"][str(length)] = {
                "prompt_ids": prompt,
                "continuation_ids": continuation,
                "continuation_text": tokenizer.decode(continuation),
            }

            prompt_t = torch.tensor([prompt], device=DEVICE)
            cont_t = torch.tensor([continuation], device=DEVICE)
            baseline = run_request(model, prompt_t, cont_t, "uncached", keep_logits=True)
            assert len(baseline.logits) == DECODE_STEPS + 1, "prefill + one logit vector per decode step"
            predicted = [int(l.argmax()) for l in baseline.logits[:-1]]
            agreement = sum(p == c for p, c in zip(predicted, continuation)) / DECODE_STEPS
            manifest["workloads"][str(length)]["baseline_argmax_agreement"] = agreement
            print(f"P={length}: uncached baseline ran 1 prefill + {DECODE_STEPS} steps; "
                  f"argmax agreement with recorded continuation {agreement:.0%}")

    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
