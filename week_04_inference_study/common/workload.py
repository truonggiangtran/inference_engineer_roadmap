"""Fixed workloads: exact-length prompts and recorded continuations (created on Day 22)."""

import json
from pathlib import Path

import torch

WEEK_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = WEEK_DIR / "day_22_workloads_and_baseline" / "results" / "workload_manifest.json"
PROMPT_LENGTHS = (128, 512)
DECODE_STEPS = 32


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run Day 22 first to create the workload manifest.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_workloads(device: torch.device, path: Path = MANIFEST_PATH) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return {prompt_length: (prompt_ids [1, P], continuation_ids [1, DECODE_STEPS])} on `device`."""
    manifest = load_manifest(path)
    workloads = {}
    for length, entry in manifest["workloads"].items():
        prompt = torch.tensor([entry["prompt_ids"]], dtype=torch.long, device=device)
        continuation = torch.tensor([entry["continuation_ids"]], dtype=torch.long, device=device)
        assert prompt.shape[1] == int(length) and continuation.shape[1] == manifest["decode_steps"]
        workloads[int(length)] = (prompt, continuation)
    return workloads
