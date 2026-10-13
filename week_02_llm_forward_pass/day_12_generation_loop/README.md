# Day 12: Training versus Inference and a First Generation Loop

**Time:** 60 minutes

## Goal

Understand autoregressive execution and the limited training concepts needed to reason about inference.

## Knowledge

**Next-token prediction.** A causal LM is trained so that the logits at position `t` predict token `t+1`. For a sequence `[a, b, c, d]`:

| Input position | Sees | Target |
| --- | --- | --- |
| 0 | `a` | `b` |
| 1 | `a b` | `c` |
| 2 | `a b c` | `d` |

The targets are the inputs shifted left by one. The causal mask from Week 1 is what makes training on all positions at once legitimate: no position can see its own target. Passing `labels=input_ids` to a Hugging Face model applies this shift internally.

**Three phases of a model's life:**

| Phase | What changes | Data |
| --- | --- | --- |
| Pretraining | All weights, by gradient descent | Huge raw-text corpora |
| Instruction tuning | Weights, further fine-tuned | Conversations in a chat template |
| Inference | **Nothing.** The weights are fixed. | Your prompt |

During inference, only the *sequence* grows. Generating a token never updates a weight.

**Two separate inference controls:**

- `model.eval()` switches *module behavior*: dropout is disabled, and normalization uses its inference behavior. It does not affect autograd.
- `torch.inference_mode()` disables *autograd tracking*, so no graph is recorded and no memory is saved for backward. It does not change module behavior.

Use both for inference.

**The autoregressive loop (uncached).**

```text
ids = prompt
repeat n times:
    logits = model(ids).logits          # GPU: full forward over ALL current tokens
    next = logits[:, -1].argmax(-1)     # GPU: argmax over the vocabulary
    ids = cat([ids, next])              # GPU tensor op; Python decides whether to continue (CPU)
```

Each step depends on the previous token, so the steps run in sequence. Without a cache, step `k` recomputes the whole prefix, so the total work grows roughly quadratically with output length. Day 13 removes that redundancy.

The Python loop, its control flow, and kernel launches run on the CPU. The forward pass and argmax run on the GPU. Calling `.item()` or checking for EOS in Python forces a CPU–GPU synchronization at every step.

## References

- [Generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation): greedy decoding with `do_sample=False`.
- [Python control flow](https://docs.python.org/3/tutorial/controlflow.html): `for` loops.
- [`torch.inference_mode`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html): its effect and how it differs from `no_grad`.

## Practice

Complete [exercise.py](exercise.py). It uses FP32 so greedy tokens can be compared exactly.

1. `shifted_pairs()`: list the (input prefix → target) pairs for a short sequence. Compute the loss manually from shifted logits and targets, and compare it with `model(ids, labels=ids).loss`.
2. `check_inference_state()`: confirm that `model.training` is `False`, inference mode is enabled, and new tensors don't require gradients.
3. `greedy_uncached()`: implement the loop above, and compare its tokens with `model.generate(do_sample=False, use_cache=False)`.

Mark in comments which lines run on the CPU and which operate on GPU tensors.

```bash
python week_02_llm_forward_pass/day_12_generation_loop/exercise.py
```

## Done when

You can explain why generated tokens are sequentially dependent and why fixed inference weights are not updated after each generated token.
