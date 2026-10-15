"""Day 23 solution: validate cached against uncached logits before timing anything."""

import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DECODE_STEPS, load_model, load_workloads, run_request

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent / "results"
# Starting points (rtol, atol); justify or adjust from the observed error distribution.
TOLERANCES = {torch.float32: (1e-3, 1e-3), torch.bfloat16: (5e-2, 5e-1)}


def compare_logits(reference: list[torch.Tensor], candidate: list[torch.Tensor], rtol: float, atol: float) -> dict:
    assert len(reference) == len(candidate)
    max_abs, max_rel, margins = [], [], []
    agree = 0
    passed = True
    for ref, cand in zip(reference, candidate):
        diff = (cand - ref).abs()
        max_abs.append(diff.max().item())
        max_rel.append((diff / ref.abs().clamp_min(1e-6)).max().item())
        if ref.argmax() == cand.argmax():
            agree += 1
        else:
            top2 = torch.topk(ref, 2).values
            margins.append((top2[0] - top2[1]).item())
        try:
            torch.testing.assert_close(cand, ref, rtol=rtol, atol=atol)
        except AssertionError:
            passed = False
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "argmax_agreement": agree / len(reference),
        "disagreement_margins": margins,
        "passed": passed,
    }


def validate(dtype: torch.dtype, lengths: list[int]) -> list[dict]:
    model, _ = load_model(dtype, DEVICE)
    workloads = load_workloads(DEVICE)
    rtol, atol = TOLERANCES[dtype]
    reports = []
    with torch.inference_mode():
        for length in lengths:
            prompt, continuation = workloads[length]
            uncached = run_request(model, prompt, continuation, "uncached", keep_logits=True)
            cached = run_request(model, prompt, continuation, "cached", keep_logits=True)
            expected_lengths = [length + i for i in range(DECODE_STEPS + 1)]
            assert cached.cache_lengths == expected_lengths, "cache must grow by exactly one per step"

            comparison = compare_logits(uncached.logits, cached.logits, rtol, atol)
            report = {"dtype": str(dtype), "prompt_len": length, "rtol": rtol, "atol": atol,
                      "steps_compared": len(cached.logits), **comparison,
                      "worst_abs_err": max(comparison["max_abs_err"])}
            reports.append(report)
            print(f"{dtype} P={length}: passed={report['passed']} worst |err|={report['worst_abs_err']:.3e} "
                  f"argmax agreement={report['argmax_agreement']:.0%}")
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return reports


def main() -> None:
    reports = validate(torch.float32, [128])
    if DEVICE.type == "cuda":
        reports += validate(torch.bfloat16, [128, 512])
    else:
        print("CPU only: skipping the BF16 benchmark-dtype check; results are not GPU validations.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "cache_correctness.json").write_text(json.dumps(reports, indent=2))
    failed = [r for r in reports if not r["passed"]]
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed: investigate before timing (see README).")
    print("All cache correctness checks passed.")


if __name__ == "__main__":
    main()
