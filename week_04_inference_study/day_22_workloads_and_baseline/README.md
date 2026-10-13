# Day 22: Prepare the Fixed Workloads and Baseline

**Time:** 60 minutes

## Goal

Construct controlled inputs and a working uncached baseline.

## Knowledge

**Exact-length prompts.** Prompt length must be a *token* count, not a character or word count. Build each prompt deterministically:

1. Tokenize a fixed seed text.
2. Repeat it until it has at least `L` tokens.
3. Truncate to exactly `L`.

Record how you did it. Use no chat template and no special tokens; the Qwen3 tokenizer adds no BOS token by default. Verify the length after building it.

**Teacher-forced benchmarking.** Live generation lets the model choose tokens. Its output can end early at EOS or vary with sampling, which changes the amount of work done. Instead, record a fixed 32-token continuation once, using greedy generation with `min_new_tokens = max_new_tokens = 32`, and then **feed those exact tokens** in every run of both modes. This is *teacher-forced inference benchmarking*: the measured computation is identical across modes and repetitions. It is not a live text-generation demonstration, so label it that way in your report.

**The uncached baseline** (in [`common/runners.py`](../common/runners.py), `mode="uncached"`):

```text
prefill: forward(prompt)                                   -> logits for continuation[0]
step i:  forward(prompt + continuation[:i+1]), no cache    -> logits for the next token
```

It uses the same settings as the cached path:

- `logits_to_keep=1`
- an all-ones mask covering past and current tokens
- the same dtype and attention backend

Read the runner and confirm that it matches your Day 21 protocol. You may adapt it if your protocol differs, but then change it for **both** modes.

**Shared code.** From today on, the day scripts import `common/`, which ensures that correctness, timing, and memory runs all exercise the same code path.

## References

- [Tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): encoding without special tokens.
- [Generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation): `min_new_tokens`, `max_new_tokens`, and `do_sample=False`.
- [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): reproducible workloads.

## Practice

Complete [exercise.py](exercise.py):

1. `build_exact_length_prompt()`: produce exactly 128 or 512 token IDs from the seed text.
2. `record_continuation()`: record 32 greedy continuation tokens.
3. The script saves `results/workload_manifest.json`, which Days 23–26 read. It records the model revision, seed text, method, IDs, and decoded text.
4. Run the shared uncached baseline on both workloads, confirm it produces exactly 1 + 32 logit vectors, and report how often its greedy argmax agrees with the recorded continuation. Agreement is informational, because teacher forcing does not require it.

```bash
python week_04_inference_study/day_22_workloads_and_baseline/exercise.py
```

## Done when

Both workloads run for exactly the specified steps, and your manifest distinguishes the benchmark from a live text-generation demonstration.
