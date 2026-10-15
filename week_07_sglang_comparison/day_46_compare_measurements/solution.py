"""Reject mismatched configurations and actual token workloads before comparison."""
import argparse
import json
from pathlib import Path


def validate_pair(left: dict, right: dict) -> None:
    for key in ("model", "metadata", "manifest_sha256", "max_tokens", "concurrency",
                "requests", "warmup", "temperature", "endpoint"):
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Incompatible measurements: config.{key}")
    if left["config"]["label"] == right["config"]["label"]:
        raise ValueError("Use distinct labels identifying framework and cache state.")
    signature = lambda rows: [(row["prompt_index"], row["prompt_tokens"], row["output_tokens"])
                              for row in rows]
    if signature(left["rows"]) != signature(right["rows"]):
        raise ValueError("Actual prompt/output token lengths differ; report separately, not as matched work.")


def compare(left: dict, right: dict) -> dict:
    validate_pair(left, right)
    metrics = ("ttft_median_ms", "ttft_p95_ms", "e2e_median_ms", "output_tokens_per_s")
    return {
        "left": left["config"]["label"], "right": right["config"]["label"],
        "metrics": {key: {"left": left["summary"][key], "right": right["summary"][key]}
                    for key in metrics},
        "caution": "Not a universal ranking. Verify server flags, cache state and repeated-run variance.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "comparison.json")
    args = parser.parse_args()
    report = compare(json.loads(args.left.read_text(encoding="utf-8")),
                     json.loads(args.right.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
