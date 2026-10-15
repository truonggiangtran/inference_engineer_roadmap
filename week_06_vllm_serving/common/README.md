# Shared Serving Client

Use [client.py](client.py) for **both** frameworks. It uses raw text
`/v1/completions` to avoid chat-template differences. It requires SSE streaming
usage accounting, text, a finish reason, and `[DONE]`; incomplete responses fail
the run instead of producing plausible results.

## Prepare once

Run the Day 38 manifest solution, then create a `metadata.json` in that day's
`results/` directory. Fill in **actual** values; this schema is not sample evidence:

```json
{
  "model_revision": "TODO: immutable model commit",
  "tokenizer_revision": "TODO: immutable tokenizer commit",
  "dtype": "TODO: actual dtype",
  "hardware": "TODO: GPU model and driver",
  "context_limit": "TODO: actual configured limit"
}
```

Keep these comparison metadata fields equal across systems. Put framework version,
full command, backend, memory budget, graph/cache flags, and cache-state protocol
in each run's accompanying worksheet; they intentionally can differ and must be
disclosed. Use a distinct `--label` naming the framework/configuration/cache state.

## Run from workspace root (Linux)

```bash
python week_06_vllm_serving/common/client.py \
  --manifest week_06_vllm_serving/day_38_concurrency_sweep/results/prompts.json \
  --metadata week_06_vllm_serving/day_38_concurrency_sweep/results/metadata.json \
  --label vllm-baseline-c1-warm \
  --concurrency 1 --requests 32 --warmup 3 --max-tokens 32 \
  --output week_06_vllm_serving/day_38_concurrency_sweep/results/vllm-c1.json
```

Repeat at concurrency 2, 4, and 8 with unique filenames. For SGLang, use the same
command with its base URL and label, retaining the manifest and metadata. Do not
overwrite previous results. No server is launched or stopped by this client.

## Measurement contract

- Closed-loop thread workers, no think time. This is **not** an open-loop arrival
  process and does not establish overload behavior or an SLO.
- Per-request TTFT: worker submission to first **nonempty text chunk** received by
  the client. Includes HTTP, queueing and server execution; not isolated GPU prefill.
- E2E: worker submission to stream completion. Client executor queue time is excluded.
- Aggregate output tokens/s: server-reported completion tokens divided by whole-run
  elapsed time, excluding warmup but including worker startup and drain.
- Chunk arrival gaps are **not true ITL**: chunks can contain multiple tokens.
  Post-first-chunk time per remaining token is a labeled approximation including
  terminal overhead. Use server-native per-token instrumentation for real ITL.
- EOS can shorten output. Inspect actual prompt/output lengths before comparisons.
- Warmup can warm prefix caches. For cold-state experiments set `--warmup 0`,
  restart or use a documented cache reset, and distinguish first request from later
  reuse within a run. Do not call an entire shared-prefix run "all cold".
- Connection failures, malformed JSON, missing usage, and server errors propagate.
  No failed request is silently dropped from a throughput denominator.
- Use large enough repeated runs for distribution analysis. A 32-request sample
  p95 is descriptive only; do not present it as a stable tail estimate.

Only synthetic prompts are intended. HTTP destinations are restricted to localhost.
