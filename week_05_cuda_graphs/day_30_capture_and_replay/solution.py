"""Day 30: compare setup cost with steady-state eager/replay execution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import compare, require_cuda, save

if __name__ == "__main__" and require_cuda():
    save(Path(__file__).parent / "results", "capture.json", compare(4096))
