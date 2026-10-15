"""Day 33: one retained graph per bucket; pointwise padding is safe here only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda, save


def choose_bucket(length: int, buckets: tuple[int, ...]) -> int:
    if length <= 0:
        raise ValueError("Length must be positive.")
    for size in sorted(buckets):
        if length <= size:
            return size
    raise ValueError("No bucket can hold this input.")


def main() -> None:
    if not require_cuda():
        return
    baseline = torch.cuda.memory_allocated()
    cases = {size: GraphCase(size) for size in (256, 1024, 4096)}
    for case in cases.values():
        case.capture()
    retained_delta = torch.cuda.memory_allocated() - baseline
    rows = []
    for length in (200, 700, 3500):
        bucket = choose_bucket(length, tuple(cases))
        case = cases[bucket]
        values = torch.randn(length, device="cuda")
        case.x.zero_()
        case.x[:length].copy_(values)
        expected = case.eager()[:length].clone()
        case.replay()
        torch.testing.assert_close(case.output[:length], expected)
        rows.append({"length": length, "bucket": bucket, "padding_elements": bucket - length})
    save(Path(__file__).parent / "results", "buckets.json", {
        "rows": rows,
        "retained_allocated_delta_bytes": retained_delta,
        "note": "Includes buffers and graph pools; not total device memory.",
    })


if __name__ == "__main__":
    main()
