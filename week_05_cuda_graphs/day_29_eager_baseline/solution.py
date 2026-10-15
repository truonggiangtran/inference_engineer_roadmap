"""Day 29: synchronized eager baseline; buffers are preallocated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, measure, require_cuda, save


def main() -> None:
    if require_cuda():
        case = GraphCase(4096)
        save(Path(__file__).parent / "results", "eager.json", measure(case.eager))


if __name__ == "__main__":
    main()
