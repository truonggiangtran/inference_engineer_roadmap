"""Model loading and environment capture for the capstone study."""

import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(dtype: torch.dtype = torch.bfloat16, device: torch.device | None = None,
               revision: str | None = None, attn_implementation: str = "sdpa"):
    """Load tokenizer and model in eval mode on `device` with an explicit dtype and attention backend."""
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=dtype, attn_implementation=attn_implementation
    )
    return model.to(device).eval(), tokenizer


def environment_info(model=None) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if model is not None:
        info["model_id"] = MODEL_ID
        info["model_revision"] = getattr(model.config, "_commit_hash", None)
        info["dtype"] = str(next(model.parameters()).dtype)
        info["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    return info
