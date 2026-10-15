"""Day 1 exercise: record the environment and create your first tensors."""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import transformers

import torch

RESULTS_DIR = Path(__file__).parent / "results"


def run_command(args: list[str]) -> str | None:
    """Return a command's first output line, or None if it is unavailable."""
    if shutil.which(args[0]) is None:
        return None
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[-1] if output else None


def environment_report() -> dict:
    """Collect versions and GPU details.

    TODO: return a dict containing at least these keys:
      python, platform, torch, torch_cuda_runtime, transformers, cuda_available,
      gpu_name, compute_capability, torch_arch_list, driver, nvcc
    Use None for values that are unavailable (for example, no GPU).
    Hints: torch.version.cuda, torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0), torch.cuda.get_arch_list(),
    run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    run_command(["nvcc", "--version"]).
    """

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__ if "transformers" in sys.modules else None,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "torch_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else None,
        "driver": run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "nvcc": run_command(["nvcc", "--version"]),
    }


def token_ids_to_tensor(token_ids: list[int]) -> torch.Tensor:
    """TODO: return an int64 tensor shaped [1, len(token_ids)] (a batch of one sequence)."""
    return torch.tensor([token_ids], dtype=torch.int64)


def gpu_smoke_test() -> str:
    """TODO: if CUDA is available, move a small tensor to the GPU, compute something
    (for example, x * 2 + 1), synchronize, and return a description of the result.
    Otherwise return a string beginning with "BLOCKER:" that explains why."""
    if not torch.cuda.is_available():
        return "BLOCKER: CUDA is not available."
    try:
        x = torch.arange(4, dtype=torch.float32, device="cuda")
        y = x * 2 + 1
        torch.cuda.synchronize()
        return f"Success on {torch.cuda.get_device_name(0)}: {y.cpu().tolist()}"
    except Exception as e:
        return f"BLOCKER: GPU test failed with error: {e}"


def main() -> None:
    report = environment_report()
    required = {
        "python", "platform", "torch", "torch_cuda_runtime", "transformers", "cuda_available",
        "gpu_name", "compute_capability", "torch_arch_list", "driver", "nvcc",
    }
    missing = required - report.keys()
    assert not missing, f"environment_report is missing keys: {missing}"

    ids = token_ids_to_tensor([101, 2009, 2003, 1037, 3231])
    assert ids.shape == (1, 5), f"expected shape (1, 5), got {tuple(ids.shape)}"
    assert ids.dtype == torch.int64, f"expected int64, got {ids.dtype}"
    print(f"token tensor: shape={tuple(ids.shape)} dtype={ids.dtype} device={ids.device}")

    smoke = gpu_smoke_test()
    print(f"GPU check: {smoke}")
    if report["cuda_available"] and report["torch_arch_list"] and "sm_120" not in report["torch_arch_list"]:
        print("WARNING: this PyTorch build does not list sm_120; reinstall from the official selector.")

    report["gpu_smoke_test"] = smoke
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "environment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {RESULTS_DIR / 'environment.json'}")


if __name__ == "__main__":
    main()
