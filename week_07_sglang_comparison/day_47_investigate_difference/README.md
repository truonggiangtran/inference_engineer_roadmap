# Day 47: Investigate One Difference

**Time:** 60 minutes. **Goal:** Explain an observation rather than speculate about runtime internals.

## Knowledge

An observed difference is a starting point for investigation. It might come from
scheduling, prefix hits, attention backend choice, graph capture policy, token
counts, or memory pressure. Choose one concrete result and write an alternative
explanation before collecting more data. A hypothesis becomes useful when a
small experiment could disprove it.

Startup logs identify effective execution backends and fallbacks. Runtime metrics
can reveal cache hits, active requests, and queueing. A diagnostic timeline may
show submission gaps or long prefill regions, but profiling can perturb execution.
Keep diagnostic runs separate from the unprofiled benchmark and document the
installed release's profiling procedure rather than assuming every runtime exposes
the same endpoint.

If changing a flag also changes compilation or kernel selection, record the
coupling and narrow the conclusion. If evidence is unavailable, an unresolved
explanation is preferable to a confident invented mechanism. Reproducing the
observation in reversed test order can help distinguish runtime behavior from
clock drift or interference. Make only the smallest configuration change needed
to test the chosen mechanism.

## References

- [SGLang server arguments](https://docs.sglang.io/advanced_features/server_arguments.html): logs, metrics and profiling options.
- [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): timeline interpretation.

## Practice and run

Fill in [worksheet.md](worksheet.md). Repeat the chosen shared-client condition,
collect one separate diagnostic log/profile using supported release instructions,
and run one falsification experiment. No new benchmark client is needed.

**Done when:** Your explanation has evidence, an alternative, and a stated uncertainty.
