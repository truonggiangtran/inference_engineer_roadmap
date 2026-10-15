# Day 39: Cache and Scheduler Experiments

**Time:** 60 minutes. **Goal:** Isolate prefix reuse from unrelated scheduler changes.

## Knowledge

Paged KV-cache management organizes storage for active sequence states. Prefix
caching reuses compatible previously computed prefix states across requests.
These are related but different mechanisms: turning off prefix reuse does not
mean disabling every request's autoregressive KV cache.

Chunked prefill limits or splits prefill work so it can be interleaved with other
requests. Changing it can affect first-output latency and ongoing decode progress.
If you also change prefix caching, graph execution, and concurrency, you cannot
attribute the resulting difference to one mechanism.

Choose one feature and use a documented release-specific toggle. Restart between
configurations and record warmup behavior. A shared-prefix run beginning with an
empty cache becomes warm as requests populate it; it is not entirely a cold-cache
run. Compare corresponding first requests and later reuse phases separately.
Record actual cache counters if available, and do not infer a cache hit solely
from a short latency. Model execution kernels, batching, and clocks may also change
latency. A clean experiment can end with inconclusive evidence if the workload
does not meaningfully exercise the selected feature.

## References

- [vLLM prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/): reuse behavior.
- [Serve arguments](https://docs.vllm.ai/en/stable/cli/serve/): installed-release toggles.

## Practice and run

Fill in [worksheet.md](worksheet.md). Reuse the Day 38 manifest or generate the
[Day 45 prefix manifests](../../week_07_sglang_comparison/day_45_prefix_reuse/solution.py).
Run the [shared client](../common/README.md), changing only the chosen feature.
Use `--warmup 0` for a first-request cold-state test after a restart.

**Done when:** Two configurations have an explicit one-factor difference and a documented cache-state protocol.
