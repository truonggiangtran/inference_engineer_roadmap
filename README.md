# Inference Engineer Practice Workspace

Hands-on folders for the 49-day plan in [inference_engineer_roadmap.md](inference_engineer_roadmap.md).
Weeks 1–4 build the foundation; Weeks 5–7 extend it to CUDA Graphs and serving systems.

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

Weeks 5–7 include experiment/protocol worksheets where running and explaining a system
is the exercise. These have no invented measurement answers. Coding days retain separate
starters and solutions. CUDA Graph labs require CUDA and explicitly skip without it;
serving clients require a separately running local GPU server, not a CPU fallback.

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

### [Week 5: CUDA Graphs](week_05_cuda_graphs)

- [Day 29: Eager launch baseline](week_05_cuda_graphs/day_29_eager_baseline)
- [Day 30: Capture and replay](week_05_cuda_graphs/day_30_capture_and_replay)
- [Day 31: Static buffers and correctness](week_05_cuda_graphs/day_31_static_buffers)
- [Day 32: Workload size sweep](week_05_cuda_graphs/day_32_size_sweep)
- [Day 33: Shape buckets](week_05_cuda_graphs/day_33_shape_buckets)
- [Day 34: Profile and explain](week_05_cuda_graphs/day_34_profile_and_explain)
- [Day 35: Graph review](week_05_cuda_graphs/day_35_graph_review)

### [Week 6: vLLM Serving](week_06_vllm_serving)

- [Shared streaming benchmark client](week_06_vllm_serving/common)
- [Day 36: Isolated server setup](week_06_vllm_serving/day_36_server_setup)
- [Day 37: Streaming and metrics](week_06_vllm_serving/day_37_streaming_metrics)
- [Day 38: Concurrency sweep](week_06_vllm_serving/day_38_concurrency_sweep)
- [Day 39: Cache and scheduler experiments](week_06_vllm_serving/day_39_cache_and_scheduler)
- [Day 40: Tail latency and load](week_06_vllm_serving/day_40_tail_latency)
- [Day 41: Graph execution ablation](week_06_vllm_serving/day_41_graph_ablation)
- [Day 42: Serving report](week_06_vllm_serving/day_42_serving_report)

### [Week 7: SGLang Comparison](week_07_sglang_comparison)

- [Day 43: SGLang setup](week_07_sglang_comparison/day_43_server_setup)
- [Day 44: Matched workload](week_07_sglang_comparison/day_44_matched_workload)
- [Day 45: Prefix reuse](week_07_sglang_comparison/day_45_prefix_reuse)
- [Day 46: Compare measurements](week_07_sglang_comparison/day_46_compare_measurements)
- [Day 47: Investigate one difference](week_07_sglang_comparison/day_47_investigate_difference)
- [Day 48: Reproduction report](week_07_sglang_comparison/day_48_reproduction_report)
- [Day 49: Final review](week_07_sglang_comparison/day_49_final_review)

## Serving environments (Weeks 6–7)

Use **separate native Linux environments** for vLLM and SGLang. Do not add either
framework to the foundation's [requirements.txt](requirements.txt): they select their
own compatible PyTorch/CUDA dependencies. Follow each release's GPU installation
instructions, check Blackwell `sm_120` support, and record exact installed versions.
See the [vLLM setup](week_06_vllm_serving/day_36_server_setup/README.md) and
[SGLang setup](week_07_sglang_comparison/day_43_server_setup/README.md).
Freeze each working environment to its day's `results/environment.txt`.
The shared HTTP client uses only Python's standard library.

Run only one server at a time, bound to `127.0.0.1`. Use synthetic public prompts,
not company data. Record the model/tokenizer revision, dtype, server command, GPU,
driver, context limit, cache/graph settings, request manifest and output lengths.
Do not commit downloaded weights, credentials, or private prompts.
Small-model results on one GPU do not establish a universal framework ranking.

## Validate the extension

```bash
python -m unittest discover -s tests -p test_roadmap_extension.py -v
```

These CPU-safe tests check the streaming client using a temporary localhost fake
server, explicit failure paths, comparison gates, day structure and local links.
They do **not** validate actual vLLM/SGLang compatibility or CUDA Graph execution.
Run the GPU labs and real server protocols on the target Linux machine before
making performance claims.
