"""Day 25 solution: measure memory per condition and capture separate diagnostic traces."""

import gc
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DECODE_STEPS, PROMPT_LENGTHS, cache_bytes, environment_info, kv_bytes_estimate,
                    load_model, load_workloads, run_request)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
MODES = ("uncached", "cached")
RESULTS_DIR = Path(__file__).parent / "results"
MIB = 1024**2


def measure_memory(model, prompt, continuation, mode: str) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    result = run_request(model, prompt, continuation, mode, keep_cache=True)
    torch.cuda.synchronize()
    row = {
        "baseline_mib": baseline / MIB,
        "peak_delta_mib": (torch.cuda.max_memory_allocated() - baseline) / MIB,
        "retained_delta_mib": (torch.cuda.memory_allocated() - baseline) / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "cache_mib": cache_bytes(result.cache, model.config.num_hidden_layers) / MIB if result.cache is not None else 0.0,
        "cache_len": result.cache.get_seq_length() if result.cache is not None else None,
    }
    del result
    return row


def diagnostic_profile(model, prompt, continuation, mode: str) -> str:
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE.type == "cuda" else [])
    first = continuation[:, :1]
    if mode == "cached":
        cache = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True,
                      logits_to_keep=1).past_key_values
        mask = torch.ones(1, prompt.shape[1] + 1, dtype=torch.long, device=DEVICE)
        step = lambda: model(input_ids=first, attention_mask=mask, past_key_values=cache,
                             use_cache=True, logits_to_keep=1)
    else:
        ids = torch.cat([prompt, first], dim=1)
        step = lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(RESULTS_DIR / f"decode_step_{mode}.pt.trace.json"))
    sort_by = "self_cuda_time_total" if DEVICE.type == "cuda" else "self_cpu_time_total"
    return prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by, row_limit=15)


def main() -> None:
    model, _ = load_model(DTYPE, DEVICE)
    workloads = load_workloads(DEVICE)
    bpe = next(model.parameters()).element_size()
    report = {"environment": environment_info(model), "conditions": [],
              "protocol": "baseline after gc + empty_cache; peak reset; prefill + 32 steps inside region; "
                          "retained read while the cache is alive"}
    RESULTS_DIR.mkdir(exist_ok=True)

    with torch.inference_mode():
        for length in PROMPT_LENGTHS:
            prompt, continuation = workloads[length]
            estimate_mib = kv_bytes_estimate(model.config, 1, length + DECODE_STEPS, bpe) / MIB
            for mode in MODES:
                row = {"prompt_tokens": length, "mode": mode, "kv_estimate_mib": estimate_mib if mode == "cached" else 0.0}
                if DEVICE.type == "cuda":
                    row.update(measure_memory(model, prompt, continuation, mode))
                    if mode == "cached":
                        assert row["cache_len"] == length + DECODE_STEPS
                        row["cache_vs_estimate"] = row["cache_mib"] / estimate_mib - 1
                report["conditions"].append(row)
                print(row)

        prompt, continuation = workloads[max(PROMPT_LENGTHS)]
        for mode in MODES:
            table = diagnostic_profile(model, prompt, continuation, mode)
            (RESULTS_DIR / f"decode_step_{mode}_top_ops.txt").write_text(table)
            print(f"\n===== one decode step, {mode}, P={max(PROMPT_LENGTHS)} =====\n{table}")

    if DEVICE.type != "cuda":
        report["note"] = "CPU only: allocator statistics unavailable; estimates only."
        print(report["note"])
    (RESULTS_DIR / "memory.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {RESULTS_DIR / 'memory.json'}")


if __name__ == "__main__":
    main()
