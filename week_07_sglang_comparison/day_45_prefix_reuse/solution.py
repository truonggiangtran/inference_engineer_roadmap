"""Generate public shared-prefix and early-diverging prompt manifests."""
import json
from pathlib import Path


def prefix_workloads(count: int = 32) -> dict[str, list[str]]:
    if count <= 0:
        raise ValueError("Count must be positive.")
    prefix = "Caching keeps reusable computation so later requests can avoid repeated work. " * 64
    return {
        "shared": [prefix + f"\nQuestion {index}: Summarize the paragraph in one sentence."
                   for index in range(count)],
        "divergent": [f"Unique document {index}.\n" + prefix
                      + f"\nQuestion {index}: Summarize the paragraph in one sentence."
                      for index in range(count)],
    }


if __name__ == "__main__":
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    for name, prompts in prefix_workloads().items():
        (results / f"{name}.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    print("Check token lengths against your context limit. Different lengths are NOT a matched ablation.")
