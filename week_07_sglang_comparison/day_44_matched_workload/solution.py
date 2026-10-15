"""Use exactly the same benchmark CLI as vLLM; only the server/config label changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week_06_vllm_serving"))
from common.client import main


def token_lengths_match(left: list[dict], right: list[dict]) -> bool:
    fields = ("prompt_index", "prompt_tokens", "output_tokens")
    signature = lambda rows: [tuple(row[field] for field in fields) for row in rows]
    return signature(left) == signature(right)


if __name__ == "__main__":
    main()
