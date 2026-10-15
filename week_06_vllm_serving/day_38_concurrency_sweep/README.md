# Day 38: Concurrency Sweep

**Time:** 60 minutes. **Goal:** Observe the throughput–latency tradeoff under closed-loop load.

## Knowledge

Continuous batching admits new requests as earlier requests finish instead of
waiting for one fixed batch to complete. Concurrency is the number of requests
that the client can have in flight, not necessarily the exact batch size of each
GPU step. Scheduling, prompt lengths, memory pressure, and output lengths influence
the runtime's actual batches.

A closed-loop client issues work as workers become available. It does not represent
an independent stream of arrivals, so it can hide overload effects and should not
be used to claim an open-loop service capacity or latency SLO. The shared client
records request latency from worker execution; time waiting in the client's
executor queue is excluded.

Use the same manifest and settings at concurrency one, two, four, and eight.
Record aggregate output token throughput and latency together: higher throughput
can coexist with worse per-request latency. Warmup and prefix caching can change
the state of repeated prompts, so control or disclose both. The manifest uses
early varying identifiers, but this is not proof that there are zero reusable
prefix blocks; inspect runtime cache metrics if making that claim.

## References

- [vLLM overview](https://docs.vllm.ai/en/stable/): continuous batching and KV management.
- [Shared client instructions](../common/README.md): metadata schema and sweep command.

## Practice and run

Complete [exercise.py](exercise.py), then generate inputs with [solution.py](solution.py):

```bash
python week_06_vllm_serving/day_38_concurrency_sweep/exercise.py
python week_06_vllm_serving/day_38_concurrency_sweep/solution.py
```

Create real metadata and run the [shared CLI](../common/README.md) at four
concurrencies with separate output filenames. Repeat each condition to inspect variance.

**Done when:** Four recorded conditions use the same manifest, with actual token lengths and honest load boundaries.
