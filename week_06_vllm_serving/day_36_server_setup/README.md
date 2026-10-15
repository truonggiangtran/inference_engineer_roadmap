# Day 36: Isolated vLLM Server Setup

**Time:** 60 minutes. **Goal:** Start a reproducible, GPU-backed localhost server.

## Knowledge

A serving runtime does more than execute a model forward pass. It receives
requests, tokenizes prompts, schedules work, allocates KV storage, selects execution
backends, and streams results. Its installation therefore includes compiled kernels
and tightly coupled PyTorch/CUDA dependencies. Installing it over the foundation
environment can break otherwise valid exercises.

Use native Linux and a fresh environment. Match the release's supported Python,
driver, CUDA, and Blackwell architecture requirements. A successful import does not
prove every attention backend supports your GPU. Record the actual selected backend
and server startup log, including any fallback or graph-capture warning.

Keep `Qwen/Qwen3-0.6B`, one GPU, BF16, and a modest context limit initially. The
memory-utilization setting reserves room for runtime/cache use and is not the
model's measured memory footprint. Bind to localhost because this lab does not
configure authentication or internet-facing service protection. Record an immutable
model/tokenizer revision and use it for both frameworks. Keep environment and
startup evidence separate from later request measurements; startup time is not
steady-state token throughput.

## References

- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/): supported stack and wheel selection.
- [Serve CLI](https://docs.vllm.ai/en/stable/cli/serve/): model revision, dtype, memory, context and host flags.

## Practice and run (Linux)

Follow the selected release's install instructions in `.venv-vllm` (not `.venv`).
For a release supporting the chosen Python, the basic flow is:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install vllm
mkdir -p week_06_vllm_serving/day_36_server_setup/results
python -m pip freeze > week_06_vllm_serving/day_36_server_setup/results/environment.txt
# Set MODEL_REVISION to the immutable commit recorded in your worksheet.
: "${MODEL_REVISION:?Set MODEL_REVISION to your recorded immutable model commit}"
vllm serve Qwen/Qwen3-0.6B --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.7
```

Run the server in the foreground in its own terminal. After successful startup:

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

Fill in [worksheet.md](worksheet.md). Stop with Ctrl+C before another server starts.
Do not change the host driver automatically if setup fails; record the blocker.

**Done when:** Readiness succeeds, the exact environment is saved, and settings/revisions are recorded.
