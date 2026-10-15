# Day 43: Isolated SGLang Setup

**Time:** 60 minutes. **Goal:** Establish a second reproducible serving environment.

## Knowledge

SGLang is another high-performance inference runtime with its own scheduler,
cache management, kernels, and execution defaults. Its radix-based prefix reuse
is an important mechanism to investigate, not proof that every workload is faster.
Framework comparisons must account for effective defaults as well as explicit flags.

Create a separate native Linux environment rather than replacing vLLM dependencies.
Follow the selected release's driver, Python, CUDA, and Blackwell instructions;
current documentation may describe a newer stack than a previously installed
foundation environment. Record the exact working release and all startup warnings.
A successful import without a real model request is insufficient validation.

Stop vLLM before starting SGLang so the two systems do not compete for GPU memory
or execution time. Use the same model and tokenizer revisions, dtype, and context
limit. Similar memory-fraction flag values do not guarantee equal effective KV
capacity: record actual cache pool sizes from logs if making memory-constrained
comparisons. Bind to localhost, test readiness, and preserve the complete launch
command. Setup results belong in the environment record rather than the benchmark.

## References

- [SGLang installation](https://docs.sglang.io/get_started/install.html): choose a release-compatible stack.
- [Server arguments](https://docs.sglang.io/advanced_features/server_arguments.html): revisions, dtype, context, memory and graph/cache settings.

## Practice and run (Linux)

Use the official installation recipe in `.venv-sglang`. For example, where the
selected release supports this stack:

```bash
python3.12 -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install uv
uv pip install --prerelease=allow sglang
mkdir -p week_07_sglang_comparison/day_43_server_setup/results
python -m pip freeze > week_07_sglang_comparison/day_43_server_setup/results/environment.txt
: "${MODEL_REVISION:?Set MODEL_REVISION to the same immutable commit as vLLM}"
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --revision "$MODEL_REVISION" --host 127.0.0.1 --port 30000 \
  --dtype bfloat16 --context-length 2048 --mem-fraction-static 0.7
```

Set `MODEL_REVISION` to the same immutable commit used for vLLM. Confirm arguments
with the installed `--help`, and verify the tokenizer resolves to the same revision.
Keep the server in a foreground terminal, then check:

```bash
curl --fail http://127.0.0.1:30000/v1/models
python week_06_vllm_serving/day_37_streaming_metrics/solution.py \
  --base-url http://127.0.0.1:30000/v1 \
  --output week_07_sglang_comparison/day_43_server_setup/results/stream.json
```

The explicit output path preserves Day 37's vLLM smoke test.
Fill in [worksheet.md](worksheet.md).

**Done when:** Readiness and one streaming request succeed with recorded revisions and effective settings.
