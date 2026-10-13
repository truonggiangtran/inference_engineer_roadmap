# Day 28: Demonstrate Understanding and Choose the Next Step

**Time:** 45 minutes

## Goal

Verify that the month produced transferable understanding, not just runnable scripts.

## Knowledge

**Explaining without notes is the test.** If you can reconstruct the forward pass and the measurement argument out loud, without reading prepared prose, you understand them well enough to apply them to a different model or GPU. Gaps that show up while you explain are the most useful output of today.

**Choose the next project from the evidence.** Choose based on the gap the study exposed, not on what sounds impressive:

| If your study showed that… | A focused next project |
| --- | --- |
| Normalization and residual adds take a visible share of decode time | Write a **fused RMSNorm + residual** kernel (Triton or CUDA), validate it against PyTorch, and measure its effect on a whole decode step |
| Attention cost grows with context and dominates at 512 tokens | **Profile attention more deeply**: compare SDPA backends, sweep context lengths, and analyze the kernels with Nsight Compute |
| Decode steps are dominated by launch and Python overhead | Run a **runtime-focused experiment**, for example CUDA graphs or `torch.compile`, or compare against a serving runtime such as vLLM using the same protocol |

Whatever you choose, keep the same discipline: a protocol first, then correctness, then timing, with honest conclusions.

**On portability.** Don't make ROCm or AMD performance claims until you have run and validated the code on AMD hardware. Code that compiles for AMD is not the same as measured AMD results.

## References

- Your completed practice documents.
- Your benchmark protocol (Day 21).
- Your capstone report (Day 27).

Revisit the source references only *after* the self-check.

## Practice

Fill in [worksheet.md](worksheet.md). There is no model answer for this day, because the review and the next-step decision depend on your own study.

## Done when

You can explain the complete forward pass and your measured tradeoffs, identify remaining gaps, and justify a focused next project.
