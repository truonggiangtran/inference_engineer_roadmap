"""Day 25 exercise: measure memory per condition and capture separate diagnostic traces."""

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
    """TODO (CUDA only): follow the README protocol for one request and return, in MiB:
      {"baseline_mib", "peak_delta_mib", "retained_delta_mib", "reserved_mib", "cache_mib", "cache_len"}
    Use run_request(..., keep_cache=True) so the cache is alive when you read memory_allocated().
    cache_mib = cache_bytes(result.cache, num_layers) / MIB for cached mode, else 0."""
    raise NotImplementedError("TODO: implement measure_memory")


def diagnostic_profile(model, prompt, continuation, mode: str) -> str:
    """TODO: profile ONE decode step at this prompt length, separately from timing runs.
      cached:   prefill the prompt (use_cache=True) OUTSIDE the profiler, then profile one forward of
                continuation[:, :1] with past_key_values and an all-ones mask of length P + 1.
      uncached: profile one forward of cat([prompt, continuation[:, :1]]) with use_cache=False.
    Use activities CPU (+ CUDA), record_shapes=True, profile_memory=True; use logits_to_keep=1.
    Export the trace to RESULTS_DIR / f"decode_step_{mode}.pt.trace.json" and return the
    key_averages(group_by_input_shape=True) table sorted by self CUDA (or CPU) time, 15 rows."""
    raise NotImplementedError("TODO: implement diagnostic_profile")


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
