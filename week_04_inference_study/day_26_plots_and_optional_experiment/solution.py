"""Day 26 solution: plots and a results table from the Day 23-25 measurements."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DECODE_STEPS, PROMPT_LENGTHS, WEEK_DIR

RESULTS_DIR = Path(__file__).parent / "results"
TIMINGS = WEEK_DIR / "day_24_timing_results" / "results" / "timings.json"
MEMORY = WEEK_DIR / "day_25_memory_and_interpretation" / "results" / "memory.json"
CORRECTNESS = WEEK_DIR / "day_23_cached_path_validation" / "results" / "cache_correctness.json"
MODES = ("uncached", "cached")
COLORS = {"uncached": "tab:orange", "cached": "tab:blue"}


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run that day's script first.")
    return json.loads(path.read_text())


def grouped_bars(values: dict, errors: dict | None, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.38
    positions = range(len(PROMPT_LENGTHS))
    for offset, mode in zip((-width / 2, width / 2), MODES):
        xs = [p + offset for p in positions]
        yerr = None
        if errors is not None:
            yerr = [[low for low, _ in errors[mode]], [high for _, high in errors[mode]]]
        ax.bar(xs, values[mode], width, yerr=yerr, capsize=4, label=mode, color=COLORS[mode])
    ax.set_xticks(list(positions), [f"P={p}" for p in PROMPT_LENGTHS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table(timings: dict, memory: dict | None, correctness: list | None) -> str:
    dtype = timings["environment"].get("dtype", "n/a")
    header = ("| Prompt tokens | Cache mode | Dtype | Prefill median (ms) | Decode median (ms/step) | Decode tokens/s "
              "| Peak allocated memory (MiB) | Variability (decode p10-p90 ms/step) | Correctness |")
    lines = [header, "| " + " | ".join(["---"] * 9) + " |"]
    for length in PROMPT_LENGTHS:
        for mode in MODES:
            c = condition(timings, length, mode)
            peak = "n/a"
            if memory:
                row = next((r for r in memory["conditions"] if r["prompt_tokens"] == length and r["mode"] == mode), None)
                if row and "peak_delta_mib" in row:
                    peak = f"{row['peak_delta_mib']:.1f}"
            check = "n/a"
            if mode == "uncached":
                check = "ref"
            elif correctness:
                match = next((r for r in correctness if r["prompt_len"] == length and r["dtype"] == dtype), None)
                if match:
                    check = "pass" if match["passed"] else "fail"
            decode = c["decode_total_ms"]
            lines.append(
                f"| {length} | {'On' if mode == 'cached' else 'Off'} | {dtype} | {c['prefill_ms']['median']:.2f} "
                f"| {c['decode_ms_per_step']:.3f} | {c['decode_tokens_per_s']:.1f} | {peak} "
                f"| {decode['p10'] / DECODE_STEPS:.3f}-{decode['p90'] / DECODE_STEPS:.3f} | {check} |"
            )
    return "\n".join(lines)


def condition(timings: dict, length: int, mode: str) -> dict:
    return next(c for c in timings["conditions"] if c["prompt_tokens"] == length and c["mode"] == mode)


def main() -> None:
    timings = load(TIMINGS)
    memory = load(MEMORY) if MEMORY.exists() else None
    correctness = load(CORRECTNESS) if CORRECTNESS.exists() else None
    env = timings["environment"]
    context = f"{env.get('gpu_name') or 'CPU'}, {env.get('dtype')}, batch 1, {DECODE_STEPS} teacher-forced steps, n={timings['repeats']}"
    RESULTS_DIR.mkdir(exist_ok=True)

    per_step = {m: [condition(timings, p, m)["decode_ms_per_step"] for p in PROMPT_LENGTHS] for m in MODES}
    per_step_err = {m: [((c := condition(timings, p, m))["decode_ms_per_step"] - c["decode_total_ms"]["p10"] / DECODE_STEPS,
                         c["decode_total_ms"]["p90"] / DECODE_STEPS - c["decode_ms_per_step"]) for p in PROMPT_LENGTHS] for m in MODES}
    grouped_bars(per_step, per_step_err, "ms per decode step (median, p10-p90)",
                 f"Decode time per step\n{context}", RESULTS_DIR / "decode_ms_per_step.png")

    prefill = {m: [condition(timings, p, m)["prefill_ms"]["median"] for p in PROMPT_LENGTHS] for m in MODES}
    prefill_err = {m: [((c := condition(timings, p, m)["prefill_ms"])["median"] - c["p10"], c["p90"] - c["median"])
                       for p in PROMPT_LENGTHS] for m in MODES}
    grouped_bars(prefill, prefill_err, "prefill latency, ms (median, p10-p90)",
                 f"Prefill latency\n{context}", RESULTS_DIR / "prefill_ms.png")

    if memory and all("peak_delta_mib" in r for r in memory["conditions"]):
        peak = {m: [next(r for r in memory["conditions"] if r["prompt_tokens"] == p and r["mode"] == m)["peak_delta_mib"]
                    for p in PROMPT_LENGTHS] for m in MODES}
        grouped_bars(peak, None, "peak allocated above baseline, MiB",
                     f"Peak allocated memory per request (single run)\n{context}", RESULTS_DIR / "peak_memory.png")
    else:
        print("No CUDA memory statistics found; skipping the memory plot.")

    table = results_table(timings, memory, correctness)
    (RESULTS_DIR / "results_table.md").write_text(table + "\n")
    print(table)
    print(f"Saved plots and table in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
