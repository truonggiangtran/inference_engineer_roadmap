"""Day 37: send one raw completion and retain the original streaming observations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.client import stream_request


def chunk_gaps(arrivals_ms: list[float]) -> list[float]:
    if any(value < 0 for value in arrivals_ms) or any(
        b < a for a, b in zip(arrivals_ms, arrivals_ms[1:])
    ):
        raise ValueError("Arrivals must be nonnegative and ordered.")
    return [b - a for a, b in zip(arrivals_ms, arrivals_ms[1:])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "stream.json")
    args = parser.parse_args()
    row = stream_request(args.base_url, "Qwen/Qwen3-0.6B", "The capital of France is", 32)
    assert 0 < row["ttft_ms"] <= row["e2e_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
