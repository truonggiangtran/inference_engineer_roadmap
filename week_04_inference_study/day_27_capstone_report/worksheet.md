# KV Cache Inference Study: Qwen3-0.6B on an RTX 5070

## Summary

In two or three sentences, state what you compared, how you compared it, and what you found.

> 

## Question

Under identical teacher-forced workloads, how do cached and uncached decoding compare in latency and GPU memory at prompt lengths of 128 and 512 tokens?

## Setup

| Item | Value |
| --- | --- |
| GPU / VRAM / driver | |
| OS / Python | |
| PyTorch (+ CUDA runtime) / Transformers | |
| Model / revision | `Qwen/Qwen3-0.6B` @ |
| Dtype / attention backend | |
| Workload | batch 1; prompts of exactly 128 and 512 tokens; 32 teacher-forced decode steps |
| Output-logit policy | `logits_to_keep=1` in both modes |

Full protocol: [Day 21 protocol](../../week_03_gpu_execution_and_measurement/day_21_benchmark_protocol/worksheet.md)

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
# install a CUDA-enabled PyTorch for sm_120, then:
pip install -r requirements.txt
python week_04_inference_study/day_22_workloads_and_baseline/solution.py      # workload manifest
python week_04_inference_study/day_23_cached_path_validation/solution.py      # correctness gate
python week_04_inference_study/day_24_timing_results/solution.py              # timings
python week_04_inference_study/day_25_memory_and_interpretation/solution.py   # memory + traces
python week_04_inference_study/day_26_plots_and_optional_experiment/solution.py  # plots + table
```

Replace `solution.py` with `exercise.py` if you ran your own implementations.

## Model and forward pass

Summarize the forward pass in a few lines, mark which choices are specific to Qwen3, and link to the Day 8 anatomy JSON and the Day 14 walkthrough.

> 

## Correctness

| Dtype | Prompt | rtol / atol | Worst abs error | Argmax agreement | Result |
| --- | --- | --- | --- | --- | --- |
| FP32 | 128 | | | | |
| BF16 | 128 | | | | |
| BF16 | 512 | | | | |

Explain why you chose those tolerances.

> 

## Results

Paste `day_26_.../results/results_table.md` here and embed the plots.

## Findings

Write three findings, each with its evidence and its evidence level (observed, derived, estimated, or hypothesis).

1. **Finding:**
   Evidence:
2. **Finding:**
   Evidence:
3. **Finding:**
   Evidence:

## A simple expectation that was incomplete, or evidence that was inconclusive

> 

## Bottleneck hypothesis

- **Hypothesis:**
- **Supporting evidence (trace, Day 18 measurements):**
- **What would disprove it:**

## Limitations

- Teacher-forced model inference, not production serving.
- Single consumer GPU, small dense model, batch size 1.
- Timings include Python launch gaps captured on the event timeline.
- Other:

## Next experiment

Name the smallest experiment that would test your strongest remaining hypothesis.

> 

## Optional stretch experiment

If you attempted one, record the factor you changed, the correctness check, the result, and the tradeoff.

> 
