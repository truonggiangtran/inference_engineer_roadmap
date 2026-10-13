# Inference Engineer Practice Workspace

Hands-on folders for the 28-day plan in [inference_engineer_roadmap.md](inference_engineer_roadmap.md).

## Setup (native Linux)

```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build for sm_120 from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Without a GPU, the toy exercises still run on CPU. CPU results are labeled as CPU results and are not GPU measurements.

## How each day works

Each day folder contains:

- `README.md`: the day's goal, background knowledge, references, practice tasks, and completion check.
- `exercise.py`, or `exercise.cu` on Day 20: a starter file with `TODO`s and built-in checks that fail until you complete it.
- `solution.py`, or `solution.cu` on Day 20: a reference solution. Try the exercise first.
- `worksheet.md`: review and report days use this instead of code. Days 7 and 14 also include a `solution.md`.
- `results/`: created when you run the code. Commit these outputs because the capstone collects them. Large profiler traces are ignored by git.

Run a script from the repository root, for example:

```bash
python week_01_tokens_and_attention/day_01_environment_and_first_inputs/exercise.py
```

## Weeks

### [Week 1: Tokens and Attention](week_01_tokens_and_attention)

- [Day 1: Environment and first model inputs](week_01_tokens_and_attention/day_01_environment_and_first_inputs)
- [Day 2: Tokenization and Python containers](week_01_tokens_and_attention/day_02_tokenization_and_containers)
- [Day 3: Embeddings and shapes](week_01_tokens_and_attention/day_03_embeddings_and_shapes)
- [Day 4: Q/K/V projections and heads](week_01_tokens_and_attention/day_04_qkv_projections_and_heads)
- [Day 5: Scaled dot products, masking, and softmax](week_01_tokens_and_attention/day_05_scaled_dot_product_and_softmax)
- [Day 6: Attention function validation](week_01_tokens_and_attention/day_06_attention_function_validation)
- [Day 7: Week 1 review](week_01_tokens_and_attention/day_07_week1_review)

### [Week 2: The Complete LLM Forward Pass](week_02_llm_forward_pass)

- [Day 8: Inspect a real decoder model](week_02_llm_forward_pass/day_08_inspect_decoder_model)
- [Day 9: RoPE and grouped-query attention](week_02_llm_forward_pass/day_09_rope_and_gqa)
- [Day 10: Normalization and residuals](week_02_llm_forward_pass/day_10_normalization_and_residuals)
- [Day 11: MLP, logits, and token choice](week_02_llm_forward_pass/day_11_mlp_logits_and_token_choice)
- [Day 12: Generation loop](week_02_llm_forward_pass/day_12_generation_loop)
- [Day 13: Prefill and cached decoding](week_02_llm_forward_pass/day_13_prefill_and_cached_decoding)
- [Day 14: Forward pass review](week_02_llm_forward_pass/day_14_forward_pass_review)

### [Week 3: GPU Execution and Measurement](week_03_gpu_execution_and_measurement)

- [Day 15: Map operations to GPU work](week_03_gpu_execution_and_measurement/day_15_operations_to_gpu_work)
- [Day 16: GPU timing and warmup](week_03_gpu_execution_and_measurement/day_16_gpu_timing_and_warmup)
- [Day 17: Profile prefill and decode](week_03_gpu_execution_and_measurement/day_17_profile_prefill_and_decode)
- [Day 18: Compute, bandwidth, and launch overhead](week_03_gpu_execution_and_measurement/day_18_compute_bandwidth_overhead)
- [Day 19: Model and cache memory](week_03_gpu_execution_and_measurement/day_19_model_and_cache_memory)
- [Day 20: CUDA kernel benchmark](week_03_gpu_execution_and_measurement/day_20_cuda_kernel_benchmark)
- [Day 21: Benchmark protocol](week_03_gpu_execution_and_measurement/day_21_benchmark_protocol)

### [Week 4: Reproducible Inference Study](week_04_inference_study)

- [Shared benchmark code](week_04_inference_study/common)
- [Day 22: Workloads and baseline](week_04_inference_study/day_22_workloads_and_baseline)
- [Day 23: Cached path validation](week_04_inference_study/day_23_cached_path_validation)
- [Day 24: Timing results](week_04_inference_study/day_24_timing_results)
- [Day 25: Memory and interpretation](week_04_inference_study/day_25_memory_and_interpretation)
- [Day 26: Plots and optional experiment](week_04_inference_study/day_26_plots_and_optional_experiment)
- [Day 27: Capstone report](week_04_inference_study/day_27_capstone_report)
- [Day 28: Final review](week_04_inference_study/day_28_final_review)
