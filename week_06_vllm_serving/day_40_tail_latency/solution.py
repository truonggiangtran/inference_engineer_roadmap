"""Inspect all request samples; a small-run p95 is not an SLO guarantee."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.client import percentile


def describe(rows: list[dict]) -> dict:
    values = [row["ttft_ms"] for row in rows]
    return {"n": len(values), "ttft_p50_ms": percentile(values, 0.5),
            "ttft_p95_ms": percentile(values, 0.95), "ttft_max_ms": max(values),
            "warning": "Sample quantiles only. Inspect run order, repeat runs, and EOS lengths."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "distribution.json")
    args = parser.parse_args()
    data = json.loads(args.measurement.read_text(encoding="utf-8"))
    report = describe(data["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
