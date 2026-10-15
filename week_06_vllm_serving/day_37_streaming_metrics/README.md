# Day 37: Streaming and Metrics

**Time:** 60 minutes. **Goal:** Define what the client actually observes.

## Knowledge

Streaming separates first visible output from full response completion. Time to
first token is commonly reported, but a generic HTTP client observes the first
nonempty content chunk, not necessarily the exact instant one model token was
produced. Queueing, prefill, serialization, network delivery, and buffering all
contribute to the observed value.

Server-sent events arrive as data records, with terminal usage accounting and a
completion marker. Empty records or records containing only metadata must not
be mistaken for first content. An HTTP success status alone does not establish
that generation completed; a stream can fail midway.

Token counts must come from actual server accounting rather than word counts or
the requested maximum. EOS may end generation early. A chunk can contain several
tokens or part of user-visible text, so adjacent chunk gaps are not true inter-token
latencies. The shared client preserves arrival observations and explicitly labels
its post-first-chunk token-time approximation. For precise ITL, collect server-native
token-level timestamps. Keep these distinctions when later reporting throughput
or comparing frameworks.

## References

- [vLLM compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/): completion and streaming APIs.
- [Shared measurement contract](../common/README.md): exact boundaries and limitations.

## Practice

Implement chunk-gap checks in [exercise.py](exercise.py), then inspect and run
[solution.py](solution.py) against the Day 36 server. Inspect finish reason, usage,
content chunks, and the first-content boundary. Explain why chunk gaps are not ITL.

## Run

```bash
python week_06_vllm_serving/day_37_streaming_metrics/exercise.py
python week_06_vllm_serving/day_37_streaming_metrics/solution.py
```

**Done when:** `results/stream.json` contains actual usage and correctly labeled timing observations.
