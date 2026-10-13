"""Shared Week 4 benchmark code: model loading, workloads, runners, and timing.

Days 22-26 import from here so cached and uncached paths stay identical across
correctness, timing, and memory measurements.
"""

from .model import MODEL_ID, environment_info, load_model
from .runners import PhaseTimer, RequestResult, cache_bytes, kv_bytes_estimate, layer_kv, run_request
from .workload import DECODE_STEPS, MANIFEST_PATH, PROMPT_LENGTHS, WEEK_DIR, load_manifest, load_workloads

__all__ = [
    "MODEL_ID", "environment_info", "load_model",
    "PhaseTimer", "RequestResult", "cache_bytes", "kv_bytes_estimate", "layer_kv", "run_request",
    "DECODE_STEPS", "MANIFEST_PATH", "PROMPT_LENGTHS", "WEEK_DIR", "load_manifest", "load_workloads",
]
