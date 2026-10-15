"""Local streaming completions benchmark. No third-party client dependencies."""

import argparse
import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1 or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite samples and a quantile in [0, 1] are required.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def stream_request(base_url: str, model: str, prompt: str, max_tokens: int,
                   timeout: float = 120) -> dict:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("This lab sends synthetic prompts to localhost HTTP servers only.")
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive.")
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "n": 1, "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(base_url.rstrip("/") + "/completions",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    arrivals, pieces = [], []
    usage, finish_reason = None, None
    done = False
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {error.code}: {detail}") from error
    with response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Expected an SSE stream, not a buffered JSON response.")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(f"Streaming server error: {event['error']}")
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text:
                    arrivals.append(time.perf_counter() - start)
                    pieces.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    elapsed = time.perf_counter() - start
    if not done or not arrivals or usage is None or finish_reason is None:
        raise RuntimeError("Incomplete stream: require text, usage, finish_reason and [DONE].")
    for field in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(field)) is not int or usage[field] <= 0:
            raise RuntimeError(f"Missing/invalid server token accounting: {field}")
    tokens = usage["completion_tokens"]
    return {
        "ttft_ms": arrivals[0] * 1000, "e2e_ms": elapsed * 1000,
        "prompt_tokens": usage["prompt_tokens"], "output_tokens": tokens,
        "finish_reason": finish_reason,
        "text": "".join(pieces),
        "content_chunk_arrival_ms": [value * 1000 for value in arrivals],
        "chunk_gaps_ms": [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])],
        "post_first_chunk_ms_per_remaining_token": (
            (elapsed - arrivals[0]) * 1000 / (tokens - 1) if tokens > 1 else None
        ),
    }


def summarize(rows: list[dict], elapsed_s: float) -> dict:
    if not rows or not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("Nonempty results and a positive measured duration are required.")
    return {
        "requests": len(rows), "elapsed_s": elapsed_s,
        "output_tokens_per_s": sum(row["output_tokens"] for row in rows) / elapsed_s,
        "requests_per_s": len(rows) / elapsed_s,
        "ttft_median_ms": statistics.median(row["ttft_ms"] for row in rows),
        "ttft_p95_ms": percentile([row["ttft_ms"] for row in rows], 0.95),
        "e2e_median_ms": statistics.median(row["e2e_ms"] for row in rows),
        "e2e_p95_ms": percentile([row["e2e_ms"] for row in rows], 0.95),
        "p95_caution": "Descriptive sample quantile, not an SLO guarantee; repeat larger runs.",
        "short_outputs": sum(row["output_tokens"] < row["requested_max_tokens"] for row in rows),
    }


def benchmark(base_url: str, model: str, prompts: list[str], max_tokens: int,
              concurrency: int, requests: int, warmup: int = 3,
              timeout: float = 120) -> dict:
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(p, str) and p for p in prompts
    ):
        raise ValueError("A nonempty list of nonempty text prompts is required.")
    if concurrency <= 0 or requests < concurrency or warmup < 0:
        raise ValueError("Require requests >= concurrency > 0 and warmup >= 0.")
    for index in range(warmup):
        stream_request(base_url, model, prompts[index % len(prompts)], max_tokens, timeout)

    def one(index: int) -> dict:
        offset_ms = (time.perf_counter() - start) * 1000
        row = stream_request(base_url, model, prompts[index % len(prompts)], max_tokens, timeout)
        row.update({"prompt_index": index % len(prompts), "requested_max_tokens": max_tokens,
                    "request_index": index, "request_start_offset_ms": offset_ms})
        return row

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, range(requests)))
    elapsed = time.perf_counter() - start
    return {
        "summary": summarize(rows, elapsed), "rows": rows,
        "load_model": "closed-loop threads, no think time; no open-loop arrival-rate claim",
        "latency_boundary": "worker request start to stream completion; excludes client executor queue",
        "stream_caution": "Content chunks need not equal tokens. Chunk gaps are NOT true token ITL.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Unique framework/config/cache-state label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    prompts = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    for key in ("model_revision", "tokenizer_revision", "dtype", "hardware", "context_limit"):
        if not metadata.get(key) or "TODO" in str(metadata[key]):
            raise ValueError(f"Record a real value for metadata.{key} before benchmarking.")
    canonical = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    result = benchmark(args.base_url, args.model, prompts, args.max_tokens,
                       args.concurrency, args.requests, args.warmup, args.timeout)
    result["config"] = {
        "model": args.model, "metadata": metadata, "label": args.label,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "max_tokens": args.max_tokens, "concurrency": args.concurrency,
        "requests": args.requests, "warmup": args.warmup,
        "temperature": 0, "endpoint": "raw text /v1/completions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
