"""Create a deterministic synthetic manifest; use the shared client for each load."""
import json
from pathlib import Path


def make_prompts(count: int = 32) -> list[str]:
    if count <= 0:
        raise ValueError("Count must be positive.")
    return [f"Case {index}: Explain in one sentence why caching avoids repeated work."
            for index in range(count)]


if __name__ == "__main__":
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    (results / "prompts.json").write_text(json.dumps(make_prompts(), indent=2), encoding="utf-8")
    print("Manifest written. Run the shared client at concurrency 1, 2, 4, and 8.")
