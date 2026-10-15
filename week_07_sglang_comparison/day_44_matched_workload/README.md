# Day 44: Matched Workload

**Time:** 60 minutes. **Goal:** Change the runtime, not the work or client.

## Knowledge

Two servers receiving the same text can still perform different work. Tokenizer
revisions, chat templates, special-token handling, generation defaults, EOS, and
context limits can change token counts or outputs. This study uses raw completion
prompts and one shared client to remove the chat-template and measurement-code
differences, but actual usage still needs inspection.

Keep the manifest's canonical hash, requested output limit, temperature, concurrency,
and request count equal. Record immutable revisions and effective dtype. Identical
requested limits do not imply identical completion lengths. Small numerical
differences between kernels can change greedy token choices, including EOS, even
when both systems are functioning correctly.

The comparison gate rejects runs with differing actual prompt or output lengths.
If that happens, investigate the difference or report the results as unmatched,
rather than changing a tolerance or hiding the shorter outputs. Equal lengths
are necessary for this controlled comparison but do not prove identical numerical
computation. Match cache-state procedures as well, and disclose backend defaults
that cannot be aligned. Keep separate result files for each framework and trial.

## References

- [SGLang request API](https://docs.sglang.io/basic_usage/send_request.html): compatible endpoints.
- [Shared measurement contract](../../week_06_vllm_serving/common/README.md).

## Practice and run

Complete [exercise.py](exercise.py). [solution.py](solution.py) delegates to the same
client used by vLLM; the full compatibility check is in
[Day 46](../day_46_compare_measurements/solution.py).

```bash
python week_07_sglang_comparison/day_44_matched_workload/solution.py \
  --base-url http://127.0.0.1:30000/v1 \
  --manifest week_06_vllm_serving/day_38_concurrency_sweep/results/prompts.json \
  --metadata week_06_vllm_serving/day_38_concurrency_sweep/results/metadata.json \
  --label sglang-baseline-c1-warm --concurrency 1 --requests 32 --warmup 3 \
  --output week_07_sglang_comparison/day_44_matched_workload/results/sglang-c1.json
```

Repeat concurrency 2, 4 and 8, preserving the corresponding Week 6 settings.

**Done when:** Actual workload lengths and cache procedures are checked, not merely requested limits.
