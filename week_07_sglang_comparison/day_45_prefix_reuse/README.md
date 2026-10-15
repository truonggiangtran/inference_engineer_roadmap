# Day 45: Prefix Reuse

**Time:** 60 minutes. **Goal:** Test locality and cold/warm state separately.

## Knowledge

Prefix caching reuses states for matching token prefixes. SGLang's radix-based
organization supports prefix sharing, while vLLM also supports automatic prefix
caching. A meaningful study tests workload locality and actual reuse rather than
treating one framework as cached and the other as inherently uncached.

Create shared-prefix and early-diverging prompt families. The supplied manifests
are synthetic public text, not equal-token-length proofs. Inspect server-reported
prompt lengths, ensure the context limit includes output capacity, and avoid a
causal speed claim from differing prompt lengths alone. For a cleaner ablation,
hold one manifest fixed and toggle reuse using the installed release's documented
setting.

Cold means a defined empty reuse state before a request, not simply no client
warmup. A batch of shared-prefix requests can warm itself. Restart or use a
documented reset, then distinguish first-request behavior from subsequent reuse.
For all-cold samples, reset before every measured request. Record hit counters and
cache capacity where available. Compare both frameworks within the same manifest
and protocol, rather than comparing one system's warm shared workload with the
other system's cold divergent workload.

## References

- [SGLang server arguments](https://docs.sglang.io/advanced_features/server_arguments.html): radix-cache controls.
- [vLLM prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/): corresponding mechanism.

## Practice and run

Complete [exercise.py](exercise.py), then generate inputs with [solution.py](solution.py):

```bash
python week_07_sglang_comparison/day_45_prefix_reuse/exercise.py
python week_07_sglang_comparison/day_45_prefix_reuse/solution.py
```

Use each manifest with the [shared client](../../week_06_vllm_serving/common/README.md)
on each framework, one server at a time. Use `--warmup 0` after resetting for the
first-request condition. Fill in [worksheet.md](worksheet.md).

**Done when:** You distinguish locality, cache state and actual work, supported by measured evidence.
