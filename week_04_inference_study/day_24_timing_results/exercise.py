"""Day 24 exercise: repeatable prefill and decode timing for all four conditions."""

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DECODE_STEPS, PROMPT_LENGTHS, PhaseTimer, environment_info, load_model, load_workloads, run_request

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
WARMUP, REPEATS = (3, 10) if DEVICE.type == "cuda" else (1, 3)
MODES = ("uncached", "cached")
RESULTS_DIR = Path(__file__).parent / "results"
BOUNDARIES = {
    "timer": "CUDA events on the default stream (PhaseTimer); synchronized perf_counter on CPU",
    "prefill": "event before the prompt forward -> event after it (logits_to_keep=1)",
    "decode": "event after prefill -> event after the 32nd one-token forward",
    "included": "GPU kernels and any GPU idle time caused by Python/launch overhead; mask slicing",
    "excluded": "tokenization, host-to-device copies, model loading, logits copies to CPU",
    "workload": "teacher-forced, batch 1, fresh cache per request",
}


def time_condition(model, prompt, continuation, mode: str) -> dict:
    """TODO: run WARMUP requests (record the first one's prefill_ms + decode_ms as "first_request_ms"),
    then REPEATS measured requests, each with a new PhaseTimer(DEVICE) and run_request(..., timer=timer).
    Return {"first_request_ms": float, "prefill_ms": [...], "decode_ms": [...]}."""
    raise NotImplementedError("TODO: implement time_condition")


def summarize(samples: list[float]) -> dict:
    """TODO: return {"median", "p10", "p90", "n"} (statistics.quantiles(samples, n=10) for p10/p90)."""
    raise NotImplementedError("TODO: implement summarize")


def main() -> None:
    model, _ = load_model(DTYPE, DEVICE)
    workloads = load_workloads(DEVICE)
    report = {"environment": environment_info(model), "boundaries": BOUNDARIES,
              "warmup": WARMUP, "repeats": REPEATS, "decode_steps": DECODE_STEPS, "conditions": []}

    with torch.inference_mode():
        for length in PROMPT_LENGTHS:
            prompt, continuation = workloads[length]
            for mode in MODES:
                raw = time_condition(model, prompt, continuation, mode)
                prefill, decode = summarize(raw["prefill_ms"]), summarize(raw["decode_ms"])
                condition = {
                    "prompt_tokens": length, "mode": mode, "raw": raw,
                    "prefill_ms": prefill, "decode_total_ms": decode,
                    "decode_ms_per_step": decode["median"] / DECODE_STEPS,
                    "decode_tokens_per_s": DECODE_STEPS / (decode["median"] / 1000),
                }
                report["conditions"].append(condition)
                print(f"P={length:<4} {mode:<8} prefill {prefill['median']:8.2f} ms "
                      f"[{prefill['p10']:.2f}-{prefill['p90']:.2f}]  decode {condition['decode_ms_per_step']:7.3f} ms/step "
                      f"[{decode['p10'] / DECODE_STEPS:.3f}-{decode['p90'] / DECODE_STEPS:.3f}]  "
                      f"{condition['decode_tokens_per_s']:7.1f} tok/s")

    if DEVICE.type != "cuda":
        report["note"] = "CPU only: not GPU measurements."
        print(report["note"])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "timings.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {RESULTS_DIR / 'timings.json'}")


if __name__ == "__main__":
    main()
