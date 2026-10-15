"""Capture a labeled eager/replay diagnostic trace; keep timing runs separate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GraphCase, require_cuda


def profile_case(case: GraphCase, destination: Path) -> None:
    # TODO: use CPU+CUDA activities, label both modes, synchronize, export a trace.
    raise NotImplementedError("Implement the diagnostic trace, not a speedup timer.")


if __name__ == "__main__" and require_cuda():
    case = GraphCase(4096)
    case.capture()
    case.check()
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    profile_case(case, results / "graph.trace.json")
    assert (results / "graph.trace.json").stat().st_size > 0
