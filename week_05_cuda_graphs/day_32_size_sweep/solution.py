"""Day 32: fixed launch count, changing element count."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import compare, require_cuda, save


def speedup(eager_wall_ms: float, replay_wall_ms: float) -> float:
    if eager_wall_ms <= 0 or replay_wall_ms <= 0:
        raise ValueError("Durations must be positive.")
    return eager_wall_ms / replay_wall_ms


if __name__ == "__main__" and require_cuda():
    rows = [compare(size) for size in (256, 4096, 65536, 1048576)]
    for row in rows:
        row["wall_speedup"] = speedup(row["eager"]["wall_median_ms"],
                                     row["replay"]["wall_median_ms"])
    save(Path(__file__).parent / "results", "sweep.json", rows)
