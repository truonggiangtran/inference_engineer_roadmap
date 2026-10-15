# Day 48: Reproduction Report

**Time:** 90 minutes. **Goal:** Assemble a fair, reproducible systems comparison.

## Knowledge

A strong report lets a reviewer reproduce both the numerical workload and the
operational conditions. Include environment freezes, model/tokenizer revisions,
full launch commands, effective backends, request manifests, metadata, raw
results, and analysis commands. A server's version string alone does not capture
its graph policy or cache configuration.

Separate cold start, graph setup, first-request cold-cache behavior, and steady
state. Include concurrency, real token counts, sample size, and repeated-run
variation in tables. Label client first-content latency accurately, and do not
present streamed chunk gaps as token-level ITL. Any excluded failure or aborted
run needs an explanation rather than disappearing from the evidence.

Recommendations should be workload-specific: identify which latency, throughput,
or memory objective matters and whether the observed tradeoff satisfies it.
Default-versus-default results are useful but do not isolate one algorithm.
Document at least one limitation and one negative or inconclusive observation.
The small model keeps experiments affordable but does not represent all model
sizes or distributed topologies. Keep large diagnostic traces local while
committing the reproducible protocol and compact measurement files.

## References

- Your Days 36–47 worksheets and results.
- [Client contract](../../week_06_vllm_serving/common/README.md).

## Practice and run

Complete [worksheet.md](worksheet.md). Replay one condition per framework from
your written commands, one server at a time. Confirm the comparison gate accepts
matched runs or explicitly report why it cannot.

**Done when:** A reader can reproduce the comparison and distinguish observations, hypotheses, and limitations.
