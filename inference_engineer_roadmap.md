# Inference Engineer Roadmap

## Overview


### Week 1: Build the Foundations
- **Stage goal:** Understand how tokens become tensors and how causal attention works.
- **Day 1:** Establish a reproducible environment and see how Python values become tensors.
- **Day 2:** Understand why language models consume token IDs rather than words or characters.
- **Day 3:** Understand embedding lookup and batch, sequence, and hidden dimensions.
- **Day 4:** Connect GEMM to learned query, key, and value projections.
- **Day 5:** Understand causal attention operations and their numerical concerns.
- **Day 6:** Build a clear, testable attention forward operation.
- **Day 7:** Consolidate attention knowledge and identify misunderstandings.

### Week 2: Trace the Complete Forward Pass
- **Stage goal:** Explain how a real decoder turns prompts into logits, with and without a KV cache.
- **Day 8:** Connect toy operations to a pretrained model's configuration and modules.
- **Day 9:** Understand position representation and how fewer KV heads reduce cache storage.
- **Day 10:** Understand normalization and residual connections in a decoder block.
- **Day 11:** Trace the path from decoder block to next-token prediction.
- **Day 12:** Understand autoregressive execution and inference fundamentals.
- **Day 13:** Understand cache reuse versus recomputing the prefix.
- **Day 14:** Explain the full forward pass across model concepts, tensors, and execution phases.

### Week 3: Measure GPU Execution Reliably
- **Stage goal:** Profile and benchmark prefill, decoding, and a CUDA kernel with sound methods.
- **Day 15:** Understand how Python and tensor operations map to GPU kernels.
- **Day 16:** Avoid misleading timings from asynchronous work and one-time setup.
- **Day 17:** Locate time in prefill and decode profiles without confusing them with benchmarks.
- **Day 18:** Build evidence-based compute, bandwidth, or overhead hypotheses.
- **Day 19:** Distinguish model and cache storage from temporary and allocator memory.
- **Day 20:** Apply reliable measurement to an existing CUDA kernel.
- **Day 21:** Define a fair benchmark comparison before collecting data.

### Week 4: Complete a Reproducible Inference Study
- **Stage goal:** Compare cached and uncached inference, validate correctness, and report time and memory.
- **Day 22:** Prepare controlled inputs and an uncached baseline.
- **Day 23:** Validate that caching preserves the intended computation.
- **Day 24:** Measure both cache modes under identical workloads.
- **Day 25:** Relate measured latency and memory to cached computation and context length.
- **Day 26:** Visualize results and run an optional focused experiment.
- **Day 27:** Assemble a clear, reproducible project and report.
- **Day 28:** Demonstrate transferable understanding and choose the next step.


## Daily Study Routine

The core schedule is **7 hours 15 minutes per week**, or **29 hours across 28 days**. Each week has five 60-minute sessions, one 90-minute practical session, and one 45-minute review. Use the remaining weekly budget for setup issues or unfinished exercises, not additional required topics.

For a 60-minute session, spend approximately 15 minutes on the indicated sections, 35 minutes on practice, and 10 minutes documenting results and answering the completion check. Do not read entire manuals or implement a complete LLM from scratch.

Use this sequence throughout: **model concept -> Python/PyTorch operation -> tensor shapes -> GPU work -> observed result**. Reuse one small model and one environment throughout the month.

If setup exceeds the first session, continue the initial tensor exercises on CPU and use catch-up time to resolve GPU access before the pretrained-model and profiling exercises. Do not describe CPU results as GPU measurements.

## Detailed Daily Roadmap

### Week 1: Python and PyTorch through Tokens and Attention

#### Day 1: Environment and Your First Model Inputs

**Time:** 60 minutes.

**Goal:** Establish a reproducible environment and understand how Python values become tensors.

**References:** [PyTorch installation selector][PT-INSTALL]; [Python introduction][PY-INTRO]; [PyTorch tensors][PT-TENSORS]; [NVIDIA compute capabilities][NV-GPUS]. Read installation choices, basic Python values, and tensor attributes.

**Practice document: Environment and first tensors**
- On your chosen native Linux system, establish a Python environment and install a CUDA-enabled PyTorch build that supports `sm_120`, plus Transformers. Use the official installation guidance rather than assuming a generic install command selects the right GPU build.
- Record the GPU, driver, Python, PyTorch, Transformers, and CUDA runtime versions. Record the local CUDA toolkit version if installed; PyTorch's bundled runtime and the toolkit used to compile your own kernels are different dependencies.
- Represent a few example token IDs as a Python list, convert them to an integer tensor, and inspect shape, dtype, and device. Move a small tensor to the GPU and confirm a simple operation completes.

**Done when:** You can explain a tensor's shape, dtype, and device, and have either a successful GPU check or a clearly recorded setup blocker.

#### Day 2: Tokenization, Vocabulary, and Python Containers

**Time:** 60 minutes.

**Goal:** Understand why a language model consumes token IDs rather than words or raw characters.

**References:** [Hugging Face tokenizer documentation][HF-TOKENIZER]; [Python data structures][PY-DATA]; [Qwen3 model documentation][HF-QWEN3]. Focus on encoding, decoding, special tokens, lists, and dictionaries.

**Practice document: Tokenization worksheet**
- Load the tokenizer for `Qwen/Qwen3-0.6B` without loading model weights. Encode a short sentence, a code fragment, and a sentence containing punctuation or repeated spaces.
- Record token IDs, visible token pieces, sequence lengths, and decoded text. Inspect the dictionary containing input IDs and the attention mask.
- Write a small Python function that prints these fields. Distinguish tokenization from the chat template used to format an instruction-model conversation.

**Done when:** You can explain why token count differs from word count and why the tokenizer and vocabulary must match the model.

#### Day 3: Embeddings, Shapes, Indexing, and Devices

**Time:** 60 minutes.

**Goal:** Understand an embedding lookup and track the batch, sequence, and hidden dimensions.

**References:** [PyTorch tensors][PT-TENSORS]; [Embedding API][PT-EMBEDDING]; [Illustrated GPT-2][ILLUSTRATED-GPT2]. Read the embedding discussion and tensor indexing examples; GPT-2 is a conceptual introduction, not the exact architecture of Qwen3.

**Practice document: Embedding shape worksheet**
- Create a toy embedding table with vocabulary size 16 and hidden size 8. Look up token IDs shaped `[2, 4]` and verify the output shape is `[2, 4, 8]`.
- Inspect one token's embedding, one sequence, and one batch element. Explain how indexing differs from matrix multiplication.
- Repeat the lookup on CPU and GPU, checking that indices and the embedding table are on compatible devices. Annotate every intermediate shape.

**Done when:** You can describe how integer IDs select learned vectors and account for every output dimension.

#### Day 4: Q/K/V Projections and Attention Heads

**Time:** 60 minutes.

**Goal:** Connect your GEMM knowledge to the learned projections inside attention.

**References:** [Matrix multiplication API][PT-MATMUL]; [PyTorch tensors][PT-TENSORS]; [Illustrated GPT-2][ILLUSTRATED-GPT2]. Focus on matrix multiplication, reshaping, transposing, and attention projections.

**Practice document: Projection and layout worksheet**
- Starting from a toy `[2, 4, 8]` hidden-state tensor, create query, key, and value projections using small weight matrices.
- Use two attention heads with head dimension 4. Reshape and transpose each projection into `[2, 2, 4, 4]`, identifying which axis represents heads and which represents tokens.
- Record shapes and strides before and after transposing. Explain why changing a view's layout does not necessarily copy data and why some later operations may require a contiguous layout.

**Done when:** You can connect each projection to a matrix multiplication and explain the layout transformation without confusing token and head axes.

#### Day 5: Scaled Dot Products, Masking, and Softmax

**Time:** 60 minutes.

**Goal:** Understand the mathematical operations and numerical concerns of causal attention.

**References:** [Scaled dot-product attention API][PT-SDPA]; [Softmax API][PT-SOFTMAX]; [Python control flow][PY-CONTROL]. Read the attention formula, masking rules, and function basics.

**Practice document: Causal attention worksheet**
- Compute query-key scores for the toy tensors, divide by the square root of head dimension, apply an additive causal mask, and apply softmax along the key/token dimension.
- Multiply the resulting weights by the values. Inspect one query row and explain which tokens it may attend to.
- Check that valid attention-weight rows sum approximately to one and future positions receive zero weight. Explain stable softmax conceptually by subtracting the row maximum; never make an entire row invalid in this exercise.

**Done when:** You can explain scaling, the causal mask, and the softmax axis, and distinguish attention weights from the final attention output.

#### Day 6: Build and Validate a Small Attention Function

**Time:** 90 minutes.

**Goal:** Turn the previous exercises into one understandable, testable forward operation.

**References:** [Scaled dot-product attention API][PT-SDPA]; [Tensor comparison API][PT-CLOSE]; [Python modules][PY-MODULES]. Read the reference operation's shape conventions and numerical comparison options.

**Practice document: Attention validation report**
- Organize the toy calculation into a Python function with explicit inputs and return values. Document input shapes and the purpose of each operation.
- Compare the manual output with PyTorch scaled dot-product attention using equal query/key sequence lengths, causal masking, and zero dropout. Start in FP32 and select explicit comparison tolerances.
- Test two small sequence lengths. Change only a future token and verify that an earlier token's output does not change; save the shape table and comparison results.

**Done when:** Your manual attention agrees with the reference within justified tolerances and passes the causal-independence check.

#### Day 7: Explain Attention without Looking at the Code

**Time:** 45 minutes.

**Goal:** Consolidate the week and identify misunderstandings before adding the rest of the model.

**References:** Revisit [Illustrated GPT-2][ILLUSTRATED-GPT2] and [scaled dot-product attention][PT-SDPA] only for questions you cannot answer.

**Practice document: Week-one review**
- Draw the path from token IDs through embeddings, projections, attention scores, masking, softmax, and output vectors. Annotate shapes.
- Explain why query-key multiplication is different from the final attention-value multiplication.
- List three mistakes your checks would catch, such as a wrong softmax axis, a reversed transpose, or a missing mask. Use remaining time to finish a blocked exercise.

**Done when:** You can give a five-minute explanation of causal attention and answer the week-one questions in the review worksheet below.

### Week 2: Python and PyTorch through the Complete LLM Forward Pass

#### Day 8: Inspect a Real Small Decoder-Only Model

**Time:** 60 minutes.

**Goal:** Connect the toy operations to a pretrained model's actual configuration and modules.

**References:** [Qwen3-0.6B model card][MODEL-QWEN3]; [Qwen3 documentation][HF-QWEN3]; [Qwen3-0.6B configuration][MODEL-CONFIG]. Read the model-loading example and configuration fields, not every serving-framework example.

**Practice document: Model anatomy sheet**
- Load the small model on your GPU in one explicitly selected supported dtype and inspect its module structure and configuration.
- Record layer count, hidden size, intermediate/MLP size, vocabulary size, query-head count, KV-head count, and head dimension. Derive head dimension from the configuration if it is not directly specified.
- Run a short inference example and identify one decoder block, its attention module, MLP, normalization modules, and output head. For this study, disable optional thinking behavior and keep the generation format consistent.

**Done when:** You can locate the components learned in week one inside the real model and explain why its query and KV head counts may differ.

#### Day 9: Position Information, RoPE, and Grouped-Query Attention

**Time:** 60 minutes.

**Goal:** Understand how attention represents token positions and why fewer KV heads can reduce cache storage.

**References:** [RoFormer/RoPE paper][ROPE-PAPER]; [Qwen3 documentation][HF-QWEN3]; [KV cache explanation][HF-CACHE]. Read RoPE's introductory explanation and inspect the model's attention configuration; a full derivation is optional.

**Practice document: Position and head-layout sheet**
- Contrast GPT-2's learned positional embeddings with the rotary position treatment used by the chosen model. Explain that positions are not new vocabulary tokens.
- Trace query-head and KV-head shapes for grouped-query attention. Work out how multiple query heads share KV information.
- Calculate the cache element count for a short sequence using the actual KV-head count. Explain why substituting the query-head count would overestimate storage in this model.

**Done when:** You can explain position information and grouped-query attention at the shape and memory level without claiming every decoder uses the same architecture.

#### Day 10: Normalization and Residual Connections

**Time:** 60 minutes.

**Goal:** Understand how normalization and residual additions fit into a decoder block.

**References:** [RMSNorm API][PT-RMSNORM]; [Qwen3 implementation][QWEN3-SOURCE]; [Illustrated GPT-2][ILLUSTRATED-GPT2]. Inspect the installed model implementation or the matching source version rather than assuming the repository's latest code matches your installation.

**Practice document: Normalization and residual worksheet**
- Calculate a tiny RMSNorm example using mean squared values, the epsilon term, and a scale vector. Compare with the PyTorch reference on a toy tensor.
- Follow one block's residual path and record shapes before and after each addition. Distinguish RMSNorm from LayerNorm; note the chosen model's query/key normalization where present.
- Explain which operations require reductions, which are elementwise, and why fusion might save memory traffic. Treat this as an optimization hypothesis, not a measured result.

**Done when:** You can explain normalization and residuals and identify their locations in the real block.

#### Day 11: MLP, Gating, Vocabulary Logits, and Token Choice

**Time:** 60 minutes.

**Goal:** Complete the computational path from a decoder block to the next-token prediction.

**References:** [Qwen3 implementation][QWEN3-SOURCE]; [Qwen3 configuration][MODEL-CONFIG]; [generation documentation][HF-GENERATION]. Focus on the gated MLP, output projection, and generation settings.

**Practice document: MLP and output-head sheet**
- Trace the gate, up, and down projections in the MLP. Record which dimensions expand and contract and where the activation and elementwise multiplication occur.
- Follow the final hidden state into vocabulary logits. Explain why logits are not probabilities until normalized.
- Compare greedy argmax with temperature-based sampling conceptually. Use a tiny logits vector to show how changing temperature affects its probability distribution without claiming it changes model weights.

**Done when:** You can describe the complete block and distinguish hidden states, logits, probabilities, and selected token IDs.

#### Day 12: Training versus Inference and a First Generation Loop

**Time:** 60 minutes.

**Goal:** Understand autoregressive execution and the limited training concepts needed to reason about inference.

**References:** [Generation documentation][HF-GENERATION]; [Python control flow][PY-CONTROL]; [inference mode API][PT-INFERENCE]. Read loop basics, greedy generation, and inference-only execution.

**Practice document: Generation-loop walkthrough**
- Explain next-token prediction and shifted training targets on a short token sequence. Distinguish pretraining, instruction tuning, and inference; do not build a training pipeline.
- Set the model to evaluation mode and use inference mode. Explain that evaluation behavior and disabling autograd are related but separate controls.
- Follow a short uncached greedy loop: forward the current prefix, select from its last-position logits, append a token, and repeat. Annotate which steps run on the CPU and which operate on GPU tensors.

**Done when:** You can explain why generated tokens are sequentially dependent and why fixed inference weights are not updated after each generated token.

#### Day 13: Prefill, Cached Decoding, and Cache Correctness

**Time:** 90 minutes.

**Goal:** Understand what caching reuses and how a cached forward pass differs from recomputing the prefix.

**References:** [KV cache explanation][HF-CACHE]; [cache strategies][HF-CACHE-STRATEGIES]; [generation documentation][HF-GENERATION]. Focus on the public cache API and a basic generation loop; advanced cache implementations are optional.

**Practice document: Cached-generation walkthrough**
- Process the prompt once as prefill, then forward only the new token with the previous cache. Let the model's supported API handle its cache and position conventions.
- Inspect the per-layer cached key/value shapes and track sequence growth. Reset the cache for each independent request, and ensure the attention mask covers the valid past and current inputs.
- Compare cached and uncached last-position logits on the same short token sequence. Do not blindly reuse square-attention causal masking for a single query against a longer key sequence; follow the model/API's documented mask alignment.

**Done when:** You can explain why previous keys and values are retained, why the next query is still computed, and why cached decoding still reads the valid context.

#### Day 14: Explain the Full LLM Forward Pass

**Time:** 45 minutes.

**Goal:** Produce an architecture explanation that connects the model's meaning, tensor operations, and execution phases.

**References:** Revisit [Qwen3 documentation][HF-QWEN3] and [KV cache explanation][HF-CACHE] for unresolved details.

**Practice document: Architecture walkthrough draft**
- Draw tokenizer -> embeddings -> repeated decoder blocks -> final normalization -> vocabulary logits -> token choice.
- Annotate attention, RoPE, grouped-query attention, normalization, residuals, and the gated MLP. Mark the model-specific choices rather than presenting them as universal LLM features.
- Explain the same prompt during prefill and during one cached decode step. Separate learned parameters, intermediate activations, and cached state.

**Done when:** You can give a ten-minute forward-pass explanation and answer the week-two review questions without copying the references.

### Week 3: GPU Execution, Profiling, and Reliable Measurement

#### Day 15: Map Model Operations to GPU Work

**Time:** 60 minutes.

**Goal:** Understand the boundary between Python orchestration, tensor-library calls, and GPU kernels.

**References:** [PyTorch CUDA semantics][PT-CUDA]; [CUDA best practices][CUDA-BEST]; [PyTorch profiler][PT-PROFILER]. Focus on asynchronous execution and the distinction between CPU and CUDA activities.

**Practice document: Operator-to-GPU map**
- Map projections and the MLP to matrix multiplications; map normalization, softmax, indexing, and residual additions to their operation types.
- Explain that PyTorch dispatches to implementations that may include CUDA kernels and vendor libraries, and that fusion can change the number of launches. Verify actual implementations through traces rather than assuming one kernel per Python operation.
- Define what your project will time. GPU prefill latency is not automatically end-to-end time to first token, which may also include tokenization, queuing, and token selection.

**Done when:** You can explain how high-level operations reach the GPU and distinguish CPU time, GPU timeline time, and end-to-end request latency.

#### Day 16: Correct GPU Timing and Warmup

**Time:** 60 minutes.

**Goal:** Avoid misleading timings caused by asynchronous launches and one-time initialization.

**References:** [PyTorch benchmarking][PT-BENCHMARK]; [CUDA events][PT-EVENT]; [PyTorch CUDA semantics][PT-CUDA]. Read warmup, synchronization, repeated measurements, and event timing.

**Practice document: Timing-method comparison**
- Time a small GPU operation with an unsynchronized host timer, a correctly synchronized host timer, and CUDA events. Explain why the first can measure submission rather than completed execution.
- Warm up the operation, preallocate inputs, and take repeated measurements. Keep transfers and allocation outside the timed region when measuring an isolated operation.
- Record median and spread rather than one best run. Explain that CUDA-event elapsed time measures a stream timeline interval, not necessarily the sum of only active kernel execution.

**Done when:** You can justify the timer and boundaries used for a benchmark and explain which overheads are included.

#### Day 17: Profile Prefill and Decoding Separately

**Time:** 60 minutes.

**Goal:** Identify where time goes in the model without confusing profiler output with a clean benchmark.

**References:** [PyTorch profiler][PT-PROFILER]; [generation documentation][HF-GENERATION]. Focus on CPU/CUDA activities, operator shapes, memory observations, and trace export.

**Practice document: First model profile**
- Capture a short warmed-up prefill and a short cached decode sequence, labeling the phases distinctly. Enable CPU and CUDA activities and inspect important input shapes.
- List the most expensive operators for each phase. Distinguish self time from inclusive time and avoid adding nested operator totals as if they were independent work.
- Export a trace and inspect launches, transfers, and gaps. Collect separate unprofiled timings for performance claims; shape recording and tracing add overhead.

**Done when:** You have a profile for both phases and can explain at least two observed differences using operator shapes or execution behavior.

#### Day 18: Compute, Bandwidth, and Launch Overhead

**Time:** 60 minutes.

**Goal:** Build evidence-based performance hypotheses rather than labeling every inference workload the same way.

**References:** [CUDA best practices][CUDA-BEST]; [Nsight Compute profiling guide][NSIGHT-COMPUTE]; [PyTorch CUDA semantics][PT-CUDA]. Read memory throughput, arithmetic intensity, and matrix-multiplication precision notes.

**Practice document: Bottleneck hypothesis sheet**
- Contrast a large prefill matrix multiplication with a small-batch decode operation. Relate shape, data reuse, arithmetic work, and repeated weight/cache reads.
- For one operation, estimate arithmetic work and bytes moved, listing assumptions. Explain why cache reuse, kernel fusion, and implementation details affect the estimate.
- Use the previous trace to propose a compute-, bandwidth-, or launch/CPU-overhead hypothesis. State what observation would disprove it; small models can be dominated by overhead even when a broad rule suggests otherwise.

**Done when:** You can explain a plausible bottleneck with supporting evidence and acknowledge that FP32, TF32, FP16, and BF16 do not imply identical numerical or performance behavior.

#### Day 19: Account for Model and Cache Memory

**Time:** 60 minutes.

**Goal:** Separate retained model/cache storage from temporary allocations and allocator bookkeeping.

**References:** [KV cache explanation][HF-CACHE]; [CUDA memory management][PT-MEMORY]; [Qwen3 configuration][MODEL-CONFIG]. Focus on cache dimensions and allocated versus reserved memory.

**Practice document: GPU memory worksheet**
- Estimate parameter storage using actual parameter dtypes, then estimate KV storage using layer count, batch size, sequence length, KV-head count, head dimension, and bytes per element.
- Observe allocated, reserved, and peak allocated PyTorch GPU memory around a short request. Define whether prefill is included in the peak measurement.
- Explain why retained cache memory grows with context but cached execution does not necessarily have the larger peak: uncached execution may allocate larger temporary activations. Do not equate PyTorch allocator statistics with total device usage shown by system tools.

**Done when:** You can calculate approximate KV storage and explain why estimates, PyTorch memory statistics, and total GPU usage differ.

#### Day 20: Reuse a CUDA Kernel as a Benchmarking Exercise

**Time:** 90 minutes.

**Goal:** Apply reliable measurement to code you already understand before interpreting whole-model results.

**References:** [CUDA best practices][CUDA-BEST]; [Nsight Systems user guide][NSIGHT-SYSTEMS]; [Nsight Compute profiling guide][NSIGHT-COMPUTE]. Focus on timing boundaries and tool purposes, not advanced architecture-specific tuning.

**Practice document: Existing-kernel measurement report**
- Select one existing SAXPY or transpose kernel. Validate its output against a reference using the same inputs and appropriate tolerances.
- Add or inspect warmup and repeated CUDA-event timing, keeping setup and transfers outside an isolated-kernel measurement. Test a small and a larger input and record configuration and variability.
- Obtain a basic trace or profile if your tools support the GPU. Explain the difference between Nsight Systems timeline analysis and Nsight Compute kernel metrics; do not claim occupancy or bandwidth counters were measured if the profiler could not collect them.

**Done when:** You have a trustworthy small benchmark report and can explain why its microbenchmark result does not directly establish an LLM speedup.

#### Day 21: Freeze the Capstone Benchmark Protocol

**Time:** 45 minutes.

**Goal:** Establish a fair comparison before collecting the final experiment data.

**References:** [PyTorch benchmarking][PT-BENCHMARK]; [KV cache explanation][HF-CACHE]. Revisit only the sections needed to resolve timing or cache questions.

**Practice document: Benchmark protocol draft**
- Fix model and tokenizer revisions, software versions, dtype, attention implementation, device, batch size 1, and output-logit policy. Where supported, request only last-position logits in both paths so output-head work does not change unintentionally.
- Define two exact tokenized prompt lengths, 128 and 512, and 32 decode steps. Use the same recorded continuation for both modes so early EOS or divergent sampling cannot change the amount of work.
- Define cache reset, warmup, repetitions, timing boundaries, memory boundaries, and correctness checks using the benchmark worksheet below.

**Done when:** Another person could reproduce the intended comparison from your protocol without guessing the workload or measurement boundaries.

### Week 4: Reproducible Small-LLM Inference Study

#### Day 22: Prepare the Fixed Workloads and Baseline

**Time:** 60 minutes.

**Goal:** Construct controlled inputs and a working uncached baseline.

**References:** [Tokenizer documentation][HF-TOKENIZER]; [generation documentation][HF-GENERATION]; [PyTorch benchmarking][PT-BENCHMARK]. Focus on input preparation and reproducible workloads.

**Practice document: Workload manifest**
- Prepare token sequences with exact prompt lengths of 128 and 512. Record how text was tokenized and truncated or repeated, including attention masks and special-token handling.
- Record a fixed 32-token continuation for each prompt. For the primary study, feed these continuation tokens during repeated forward passes instead of allowing sampling to change the workload; label this teacher-forced inference benchmarking.
- Implement or adapt the uncached path so each step forwards the full current prefix and selects the relevant last-position logits. Keep the same output policy and inference settings planned for the cached path.

**Done when:** Both workloads run for exactly the specified steps, and your manifest distinguishes the benchmark from a live text-generation demonstration.

#### Day 23: Validate the Cached Path before Timing It

**Time:** 60 minutes.

**Goal:** Establish that caching changes execution strategy without silently changing the intended computation.

**References:** [KV cache explanation][HF-CACHE]; [Tensor comparison API][PT-CLOSE]; [cache strategies][HF-CACHE-STRATEGIES]. Focus on cache initialization, updates, masks, and numerical validation.

**Practice document: Cache correctness report**
- Add prefill and cached one-token forward passes using the public API supported by your installed Transformers version. Start each independent run with a fresh cache.
- Compare cached and uncached last-position logits for the same prefixes and continuation tokens. Begin with a short FP32 check if necessary, then check the chosen benchmark dtype and record absolute/relative errors with justified tolerances.
- Check sequence-length growth and masks. Investigate failures instead of loosening tolerances blindly; token-level agreement alone is not a complete numerical check, and nearly tied logits can affect argmax.

**Done when:** Both execution paths pass your documented correctness criteria for the controlled inputs before speed or memory claims are made.

#### Day 24: Collect Repeatable Timing Results

**Time:** 60 minutes.

**Goal:** Measure the cache comparison under identical, explicitly defined workloads.

**References:** [PyTorch benchmarking][PT-BENCHMARK]; [CUDA events][PT-EVENT]; [PyTorch CUDA semantics][PT-CUDA]. Focus on synchronization and statistics.

**Practice document: Timing results sheet**
- For each prompt length and cache mode, perform at least three warmup requests and ten measured requests. Reset request state each time and ensure no unrelated GPU workload runs concurrently.
- Record prefill latency, total decode time, average time per decode step, and decode tokens/second. Use the same timing method and boundaries across modes; model-forward throughput is not production serving throughput.
- Report median and variability, and record cold-start or initialization separately if observed. Do not make tail-latency claims from ten samples or select only the fastest run.

**Done when:** All four primary conditions have repeat measurements and defined metrics, including exactly which CPU overheads and GPU work the timings include.

#### Day 25: Collect Memory and Explain the Results

**Time:** 60 minutes.

**Goal:** Relate observed latency and memory differences to cached computation and context length.

**References:** [CUDA memory management][PT-MEMORY]; [KV cache explanation][HF-CACHE]; [PyTorch profiler][PT-PROFILER]. Focus on memory boundaries and separately collected diagnostic traces.

**Practice document: Memory and interpretation sheet**
- Measure peak allocated memory using consistent reset points and timing/measurement boundaries. Release previous request outputs and caches before independent runs, accounting for reserved-memory reuse.
- Record retained cache storage where practical and compare it with the analytical estimate. Explain larger peaks or discrepancies instead of forcing the data to match a simple expectation.
- Capture short diagnostic traces separately from timing runs. Connect any speed difference to repeated projections/MLP work, attention shapes, cache reads, allocation behavior, or host overhead supported by the evidence.

**Done when:** You can explain both time and memory results, including a case where a simple expectation was incomplete or the evidence remained inconclusive.

#### Day 26: Create Plots and One Optional Experiment

**Time:** 60 minutes.

**Goal:** Turn raw measurements into an understandable comparison without expanding the required project.

**References:** [Matplotlib quick start][PLOT-QUICKSTART]; [PyTorch CUDA semantics][PT-CUDA]; [cache strategies][HF-CACHE-STRATEGIES]. Read only the plotting basics and any documentation needed for an optional change.

**Practice document: Results visualization sheet**
- Plot cached/uncached decode time and peak allocated memory for the two prompt lengths. Label units, workload, precision, sample count, and variability where appropriate.
- Add a table with prefill latency, decode time per step, throughput, and memory. Distinguish measured values from estimates and avoid unsupported performance extrapolations.
- Only if the required comparison is complete, try one stretch experiment: batch size 1 versus 2, or BF16 versus FP32. Change one factor at a time, repeat correctness checks, and document the additional tradeoff; otherwise use this time to improve the core results.

**Done when:** The primary study has readable plots and a results table, whether or not a stretch experiment was attempted.

#### Day 27: Assemble the Demonstrable Project

**Time:** 90 minutes.

**Goal:** Make the work understandable and reproducible for someone reviewing your engineering judgment.

**References:** [Qwen3 model card][MODEL-QWEN3]; [PyTorch benchmarking][PT-BENCHMARK]; [PyTorch profiler][PT-PROFILER]. Use these to verify attribution, configuration, and measurement descriptions.

**Practice document: Capstone report and reproduction guide**
- Assemble the environment manifest, model anatomy sheet, annotated forward pass, benchmark protocol, correctness checks, raw results, plots, and representative traces.
- Write the reproduction steps and dependencies, identifying the model/tokenizer revision and the GPU used. Explain which results are specific to the RTX 5070, chosen model size, and measurement method.
- Write three evidence-backed findings, one limitation, and one proposed next experiment. Include an honest negative or inconclusive result if that is what the measurements show; a speedup is not a required outcome.

**Done when:** A reviewer can understand the comparison, rerun it, and distinguish observed results from hypotheses without asking for missing configuration details.

#### Day 28: Demonstrate Understanding and Choose the Next Step

**Time:** 45 minutes.

**Goal:** Verify that the month produced transferable understanding, not just runnable scripts.

**References:** Your completed practice documents, benchmark protocol, and capstone report; revisit the source references only after the self-check.

**Practice document: Final review and next-step decision**
- Give a ten-minute architecture walkthrough and a five-minute performance explanation without reading prepared prose. Use the review questions below.
- Explain the cache comparison, the numerical checks, one bottleneck hypothesis, and what evidence would change your conclusion.
- Choose one next project based on the gap exposed by the study: a fused RMSNorm/residual operator, deeper attention profiling, or a runtime-focused experiment. Defer ROCm execution claims until you have access to AMD hardware for validation.

**Done when:** You can explain the complete forward pass and your measured tradeoffs, identify remaining gaps, and justify a focused next project.

## Reusable Practice Documents

### Daily Practice Note Template

Use this short template for each day. A section in one notebook or Markdown workbook is sufficient; you do not need 28 separate documents.

```text
Day and topic:
Goal in my own words:
References and sections consulted:
Inputs, shapes, dtypes, and device:
Operation or experiment performed:
Expected result and reason:
Observed result and correctness check:
GPU connection: compute, memory traffic, launches, or synchronization:
One mistake or unresolved question:
Completion check answer:
```

### Model Anatomy and Shape Worksheet

Record actual values from the chosen configuration and implementation rather than memorizing one model's dimensions.

| Component | Input shape | Output shape | Main operation | What is learned or retained? |
| --- | --- | --- | --- | --- |
| Tokenizer | Text | Token IDs and mask | Vocabulary encoding | Not a GPU model layer |
| Embedding | Batch, sequence | Batch, sequence, hidden | Table lookup | Learned embedding weights |
| Q/K/V projections | Batch, sequence, hidden | Head-specific tensors | Matrix multiplication | Learned projection weights |
| RoPE | Query/key tensors and positions | Same logical dimensions | Position-dependent rotations | Position transformation; not token IDs |
| Attention | Queries, keys, values, mask | Batch, sequence, hidden after head merge | Dot products, softmax, aggregation, output projection | Past keys and values during cached inference |
| Norm and residuals | Hidden-state tensors | Same hidden-state dimensions | Reduction, scaling, addition | Learned normalization scales where applicable |
| Gated MLP | Batch, sequence, hidden | Batch, sequence, hidden | Projections, activation, gating | Learned MLP weights |
| Output head | Final hidden states | Vocabulary logits | Matrix multiplication | Learned output weights, possibly tied to embeddings |

Account for final normalization and model-specific query/key normalization in your annotated diagram. Grouped-query attention gives queries and keys/values different head counts; do not assume identical shapes everywhere.

### Benchmark Protocol Worksheet

Complete this before collecting final results:

- **Environment:** GPU and VRAM; driver; operating system; Python; PyTorch and its CUDA runtime; Transformers; local CUDA toolkit and profiler versions if used.
- **Model:** Model/tokenizer identifiers and revisions, evaluation mode, inference mode, weight dtype, attention backend, and output-logit policy.
- **Workload:** Batch size 1; exact prompt lengths 128 and 512; fixed 32-token continuation; masks and position handling; fresh cache for every independent request.
- **Comparison:** Cached and uncached execution with identical tokens and settings. Record whether the baseline computes logits for previous positions or only the current position.
- **Correctness:** Last-position logits compared on the same sequence; explicit dtype-appropriate tolerances; cache shapes and sequence growth checked.
- **Timing:** Timer, synchronization, phase boundaries, three or more warmup requests, ten or more measured requests, and statistics. Record whether tokenization, sampling, transfers, allocation, and Python-loop overhead are included.
- **Memory:** Peak-stat reset point, whether prefill and decoding share the measurement region, allocated versus reserved memory, and cleanup between independent requests.
- **Profiling:** Separate diagnostic runs; CPU and CUDA activities; operator shapes; no profiler instrumentation in the benchmark runs used for speed claims.
- **Limitations:** Teacher-forced model inference rather than production serving; single consumer GPU; small dense model; no multi-GPU or cross-vendor performance claim.

### Results Table Template

| Prompt tokens | Cache mode | Dtype | Prefill median (ms) | Decode median (ms/step) | Decode tokens/s | Peak allocated memory (MiB) | Variability | Correctness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Off | Record actual | Measure | Measure | Calculate | Measure | Report | Record |
| 128 | On | Same | Measure | Measure | Calculate | Measure | Report | Record |
| 512 | Off | Same | Measure | Measure | Calculate | Measure | Report | Record |
| 512 | On | Same | Measure | Measure | Calculate | Measure | Report | Record |

For this batch-size-one study, average decode time per step is total decode time divided by 32; decode tokens/second is 32 divided by decode time in seconds. Label whether those durations include Python orchestration. Do not report them as a serving system's request throughput.

For a basic unquantized full-context KV cache, estimate storage as:

$$
	ext{KV bytes} \approx 2 \times \text{layers} \times \text{batch size} \times \text{cached tokens} \times \text{KV heads} \times \text{head dimension} \times \text{bytes per element}
$$

The factor of two accounts for keys and values. This estimates logical cache tensor storage, not allocator overhead, temporary activations, or all GPU usage. Sliding-window, quantized, or other specialized caches require different accounting.

### Weekly Review Questions

**Week 1**
1. Why do token count, word count, and character count differ?
2. How do token IDs become learned vectors?
3. What are the roles and shapes of queries, keys, and values?
4. Why scale attention scores, and along which dimension is softmax applied?
5. What test demonstrates that future tokens cannot influence an earlier causal-attention output?

**Week 2**
1. What does one decoder block compute, and which parts are specific to the chosen model?
2. How do RoPE, residuals, normalization, and the MLP fit into the forward pass?
3. What separates logits, probabilities, and token selection?
4. How do pretraining, instruction tuning, and inference differ?
5. What changes between prefill, uncached generation, and cached decoding?
6. Why does KV caching avoid recomputation without eliminating attention over the existing context?

**Week 3**
1. Why can an unsynchronized host timer give a misleading GPU result?
2. What do warmup, allocation, and profiler instrumentation change?
3. How do you distinguish compute limits, memory-bandwidth limits, and launch/host overhead?
4. Why do query-head count and KV-head count matter differently for cache memory?
5. How do allocated memory, reserved memory, retained cache size, and peak temporary usage differ?

**Week 4**
1. What was held constant, and what changed in your cache comparison?
2. How did you establish numerical correctness before timing?
3. What exactly do your latency and throughput numbers measure?
4. Which observation supports each performance explanation?
5. What limitation prevents generalizing the result to all models, GPUs, or serving workloads?
6. What is the smallest next experiment that could test your strongest remaining hypothesis?

## Detailed Reference Index

These are reference documents, not a requirement to complete every linked tutorial or paper. Follow the targeted reading described for each day. Match API details to the versions recorded in your environment.

### Python and PyTorch Foundations

- [Python introduction][PY-INTRO]: values, strings, lists, and basic expressions.
- [Python control flow][PY-CONTROL]: loops, conditions, and functions.
- [Python data structures][PY-DATA]: lists and dictionaries used for tokenizer/model inputs.
- [Python modules][PY-MODULES]: imports and organizing a small reusable program.
- [PyTorch installation selector][PT-INSTALL]: choose a supported GPU-enabled build.
- [PyTorch tensors][PT-TENSORS]: shapes, indexing, devices, and dtypes.
- [Embedding][PT-EMBEDDING], [matrix multiplication][PT-MATMUL], [softmax][PT-SOFTMAX], and [RMSNorm][PT-RMSNORM]: operation-specific reference pages.
- [Scaled dot-product attention][PT-SDPA]: reference attention behavior and mask semantics.
- [Inference mode][PT-INFERENCE] and [tensor comparison][PT-CLOSE]: inference execution and correctness checks.

### LLM Architecture and Model Execution

- [Illustrated GPT-2][ILLUSTRATED-GPT2]: visual decoder architecture foundation; distinguish its architecture from modern Qwen3 details.
- [RoFormer/RoPE paper][ROPE-PAPER]: introductory motivation and rotary positional information; advanced derivations are optional this month.
- [Tokenizer documentation][HF-TOKENIZER]: token encoding, decoding, special tokens, and model inputs.
- [Qwen3 model documentation][HF-QWEN3], [model card][MODEL-QWEN3], and [configuration][MODEL-CONFIG]: the chosen real model.
- [Qwen3 implementation][QWEN3-SOURCE]: inspect the corresponding installed version for module and forward-pass details.
- [Generation documentation][HF-GENERATION]: token selection, generation settings, and supported APIs.
- [KV cache explanation][HF-CACHE] and [cache strategies][HF-CACHE-STRATEGIES]: cached execution and storage conventions.

### GPU Measurement and Reporting

- [NVIDIA compute capabilities][NV-GPUS] and [CUDA 12.8 release notes][CUDA-128]: RTX 5070 architecture and the introduction of native compiler support.
- [PyTorch CUDA semantics][PT-CUDA] and [CUDA memory management][PT-MEMORY]: execution, precision, synchronization, and memory accounting.
- [PyTorch benchmarking][PT-BENCHMARK], [CUDA events][PT-EVENT], and [PyTorch profiler][PT-PROFILER]: reliable timing and diagnostic traces.
- [CUDA best practices][CUDA-BEST]: memory traffic, performance measurement, and optimization principles.
- [Nsight Systems user guide][NSIGHT-SYSTEMS] and [Nsight Compute profiling guide][NSIGHT-COMPUTE]: timeline analysis versus kernel-level analysis.
- [Matplotlib quick start][PLOT-QUICKSTART]: simple labeled result plots.

[PY-INTRO]: https://docs.python.org/3/tutorial/introduction.html
[PY-CONTROL]: https://docs.python.org/3/tutorial/controlflow.html
[PY-DATA]: https://docs.python.org/3/tutorial/datastructures.html
[PY-MODULES]: https://docs.python.org/3/tutorial/modules.html
[PT-INSTALL]: https://pytorch.org/get-started/locally/
[PT-TENSORS]: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
[PT-EMBEDDING]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
[PT-MATMUL]: https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
[PT-SOFTMAX]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
[PT-RMSNORM]: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
[PT-SDPA]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
[PT-INFERENCE]: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
[PT-CLOSE]: https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
[ILLUSTRATED-GPT2]: https://jalammar.github.io/illustrated-gpt2/
[ROPE-PAPER]: https://arxiv.org/abs/2104.09864
[HF-TOKENIZER]: https://huggingface.co/docs/transformers/main_classes/tokenizer
[HF-QWEN3]: https://huggingface.co/docs/transformers/model_doc/qwen3
[MODEL-QWEN3]: https://huggingface.co/Qwen/Qwen3-0.6B
[MODEL-CONFIG]: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
[QWEN3-SOURCE]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
[HF-GENERATION]: https://huggingface.co/docs/transformers/main_classes/text_generation
[HF-CACHE]: https://huggingface.co/docs/transformers/cache_explanation
[HF-CACHE-STRATEGIES]: https://huggingface.co/docs/transformers/kv_cache
[NV-GPUS]: https://developer.nvidia.com/cuda-gpus
[CUDA-128]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
[PT-CUDA]: https://docs.pytorch.org/docs/stable/notes/cuda.html
[PT-MEMORY]: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
[PT-BENCHMARK]: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
[PT-EVENT]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
[PT-PROFILER]: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
[CUDA-BEST]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
[NSIGHT-SYSTEMS]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
[NSIGHT-COMPUTE]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
[PLOT-QUICKSTART]: https://matplotlib.org/stable/users/explain/quick_start.html
