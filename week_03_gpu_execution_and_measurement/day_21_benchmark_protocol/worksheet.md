# Capstone Benchmark Protocol

Status: draft / frozen on `____-__-__`

## Environment

| Item | Value |
| --- | --- |
| GPU and VRAM | |
| NVIDIA driver | |
| Operating system | |
| Python | |
| PyTorch and its CUDA runtime | |
| Transformers | |
| Local CUDA toolkit / Nsight versions (if used) | |

Source: `week_01_tokens_and_attention/day_01_environment_and_first_inputs/results/environment.json`

## Model

| Item | Value |
| --- | --- |
| Model ID | `Qwen/Qwen3-0.6B` |
| Model and tokenizer revision (commit hash) | |
| Weight dtype | |
| Attention implementation | |
| `model.eval()` + `torch.inference_mode()` | yes / no |
| Output-logit policy (both modes) | |

## Workload

| Item | Value |
| --- | --- |
| Batch size | 1 |
| Prompt lengths (exact tokens) | 128, 512 |
| How prompt text was built / truncated | |
| Special tokens / chat template used? | |
| Decode steps | 32 |
| Continuation tokens: how recorded | |
| Attention mask and position handling | |
| Cache reset between requests | |

## Comparison

- What is identical between the cached and uncached modes:
- What differs:
- Does the uncached baseline compute logits for earlier positions? 

## Correctness criteria (before any timing)

| Check | Dtype | Tolerance (rtol / atol) | Justification |
| --- | --- | --- | --- |
| Last-position logits, cached vs uncached | FP32 | | |
| Last-position logits, cached vs uncached | Benchmark dtype | | |
| Cache sequence length after each step | — | exact | |
| Argmax agreement (informational) | — | — | Nearly tied logits can flip |

## Timing

| Item | Value |
| --- | --- |
| Timer | |
| Synchronization points | |
| Prefill boundary (start → end) | |
| Decode boundary (start → end) | |
| Warmup requests per condition | ≥ 3 |
| Measured requests per condition | ≥ 10 |
| Statistics reported | median, p10–p90 |
| Included: tokenization? transfers? allocation? Python loop? token selection? | |
| Other GPU workloads closed? | |

## Memory

| Item | Value |
| --- | --- |
| Peak-statistics reset point | |
| Prefill and decode in the same region? | |
| Metrics | peak allocated, retained, reserved |
| Cleanup between requests | |

## Profiling

- Separate diagnostic runs only; no profiler during the timing runs used for claims.
- Activities and options:

## Known limitations

- Teacher-forced model inference, not production serving.
- Single consumer GPU; small dense model.
- No multi-GPU or cross-vendor performance claim.
- Other:
