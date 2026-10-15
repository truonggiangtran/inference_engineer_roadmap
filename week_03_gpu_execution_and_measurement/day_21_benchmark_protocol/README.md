# Day 21: Freeze the Capstone Benchmark Protocol

**Time:** 45 minutes

## Goal

Establish a fair comparison before collecting the final experiment data.

## Knowledge

**Why freeze the protocol first?** If you decide boundaries, warmup, or workloads *after* seeing numbers, you can, unintentionally, pick the version that looks best. A protocol written in advance turns the experiment into a test rather than a search.

**What makes the cached-versus-uncached comparison fair:**

- **Identical work, except for caching.** Use the same model and tokenizer revision, dtype, attention backend, device, batch size, prompt tokens, and continuation tokens.
- **Teacher forcing.** Feed the same recorded 32-token continuation in both modes, so early EOS or divergent sampling can't change the amount of work.
- **The same output-logit policy.** If the uncached path computed logits for every position while the cached path computed only the last one, the comparison would include `lm_head` work over the whole vocabulary that the cached path doesn't do. Request `logits_to_keep=1` in both modes.
- **Fresh state.** Start every independent request with a new cache, and release previous outputs.
- **Stated boundaries.** Specify exactly what the timer and the memory region include:
  - tokenization
  - host-to-device copies
  - Python loop overhead
  - token selection

**Two workloads:** exact tokenized prompt lengths of 128 and 512, each with 32 decode steps. Comparing them shows how context length changes the gap between the two modes.

## References

- [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): warmup and repetition.
- [KV cache explanation](https://huggingface.co/docs/transformers/cache_explanation): cache reset and masks.

## Practice

Fill in [worksheet.md](worksheet.md). It follows the Benchmark Protocol Worksheet from the roadmap. Fill in every field; write "N/A" with a reason where something doesn't apply. Week 4's code (`week_04_inference_study/common/`) implements this protocol, so check that the two agree.

There is no model answer for this day, because the protocol must reflect your own environment and decisions.

## Done when

Another person could reproduce the intended comparison from your protocol without guessing the workload or measurement boundaries.
