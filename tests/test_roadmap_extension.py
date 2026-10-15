"""CPU-safe checks; the local fake server is NOT model/GPU performance evidence."""

import ast
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from week_06_vllm_serving.common import client
from week_05_cuda_graphs.common import require_cuda
from week_06_vllm_serving.common.client import benchmark, percentile, stream_request, summarize
from week_07_sglang_comparison.day_45_prefix_reuse.solution import prefix_workloads
from week_07_sglang_comparison.day_46_compare_measurements.solution import compare, validate_pair

ROOT = Path(__file__).resolve().parents[1]


class FakeServer(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        if body["prompt"] == "http-error":
            self.send_error(500, "test failure")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        records = [
            {"choices": [{"text": "", "finish_reason": None}]},
            {"choices": [{"text": "Hello", "finish_reason": None}]},
            {"choices": [{"text": " world", "finish_reason": "length"}]},
        ]
        if body["prompt"] == "server-error":
            records = [{"error": {"message": "test stream error"}}]
        elif body["prompt"] == "malformed":
            self.wfile.write(b"data: {not-json}\n\n")
            return
        elif body["prompt"] != "missing-usage":
            records.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})
        for record in records:
            self.wfile.write(("data: " + json.dumps(record) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        if body["prompt"] != "incomplete":
            self.wfile.write(b"data: [DONE]\n\n")


def fixture(label: str) -> dict:
    return {
        "config": {
            "model": "toy", "metadata": {
                "dtype": "float32", "hardware": "mock-only",
                "model_revision": "test", "tokenizer_revision": "test", "context_limit": 64,
            },
            "manifest_sha256": "test", "max_tokens": 2, "concurrency": 1,
            "requests": 1, "warmup": 0, "temperature": 0,
            "endpoint": "raw text /v1/completions", "label": label,
        },
        "rows": [{"prompt_index": 0, "prompt_tokens": 7, "output_tokens": 2}],
        "summary": {"ttft_median_ms": 1, "ttft_p95_ms": 1,
                    "e2e_median_ms": 2, "output_tokens_per_s": 10},
    }


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        if cls.thread.is_alive():
            raise RuntimeError("Fake server did not stop.")

    def test_stream_usage_and_boundaries(self) -> None:
        row = stream_request(self.url, "toy", "normal", 2)
        self.assertEqual(row["text"], "Hello world")
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(len(row["content_chunk_arrival_ms"]), 2)
        self.assertEqual(len(row["chunk_gaps_ms"]), 1)
        self.assertGreater(row["ttft_ms"], 0)
        self.assertGreaterEqual(row["e2e_ms"], row["ttft_ms"])

    def test_stream_failures_are_explicit(self) -> None:
        for prompt in ("http-error", "server-error", "missing-usage", "incomplete", "malformed"):
            with self.subTest(prompt=prompt), self.assertRaises((RuntimeError, ValueError)):
                stream_request(self.url, "toy", prompt, 2)

    def test_external_destination_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stream_request("https://example.com/v1", "toy", "synthetic", 2)

    def test_closed_loop_concurrency_and_summary(self) -> None:
        result = benchmark(self.url, "toy", ["one", "two"], 2, 2, 8, warmup=1)
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(result["summary"]["requests"], 8)
        self.assertEqual([row["request_index"] for row in result["rows"]], list(range(8)))
        self.assertGreater(result["summary"]["output_tokens_per_s"], 0)
        self.assertIn("NOT true token ITL", result["stream_caution"])

    def test_failed_request_does_not_become_success(self) -> None:
        with self.assertRaises(RuntimeError):
            benchmark(self.url, "toy", ["one", "missing-usage"], 2, 2, 4, warmup=0)

    def test_invalid_load_rejected(self) -> None:
        for prompts, concurrency, requests, warmup in (
            ([], 1, 2, 0), (["x"], 0, 2, 0), (["x"], 4, 2, 0),
            (["x"], 1, 2, -1), ({"bad": "shape"}, 1, 2, 0),
        ):
            with self.subTest(prompts=prompts, concurrency=concurrency):
                with self.assertRaises(ValueError):
                    benchmark(self.url, "toy", prompts, 2, concurrency, requests, warmup)

    def test_summary_uses_actual_tokens(self) -> None:
        rows = [{"output_tokens": 2, "requested_max_tokens": 4, "ttft_ms": 1, "e2e_ms": 3},
                {"output_tokens": 4, "requested_max_tokens": 4, "ttft_ms": 2, "e2e_ms": 5}]
        summary = summarize(rows, 2)
        self.assertEqual(summary["output_tokens_per_s"], 3)
        self.assertEqual(summary["short_outputs"], 1)

    def test_quantiles(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)
        self.assertEqual(percentile([7], 0.95), 7)
        for values, q in (([], 0.5), ([1], -1), ([float("nan")], 0.5)):
            with self.assertRaises(ValueError):
                percentile(values, q)

    def test_cli_roundtrip_and_comparison_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inference-lab-test-") as directory:
            temp = Path(directory)
            manifest, metadata = temp / "prompts.json", temp / "metadata.json"
            manifest.write_text(json.dumps(["Hello", "Public synthetic text"]), encoding="utf-8")
            metadata.write_text(json.dumps(fixture("test")["config"]["metadata"]), encoding="utf-8")
            for label in ("left", "right"):
                args = [
                    "client.py", "--base-url", self.url, "--manifest", str(manifest),
                    "--metadata", str(metadata), "--label", label,
                    "--output", str(temp / f"{label}.json"),
                    "--requests", "4", "--concurrency", "2", "--max-tokens", "2", "--warmup", "0",
                ]
                with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
                    client.main()
            script = (ROOT / "week_07_sglang_comparison" /
                      "day_46_compare_measurements" / "solution.py")
            output = temp / "comparison.json"
            result = subprocess.run([
                sys.executable, str(script), str(temp / "left.json"),
                str(temp / "right.json"), "--output", str(output),
            ], capture_output=True, text=True, check=True)
            self.assertIn("metrics", result.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["right"], "right")

    def test_placeholder_metadata_rejected_before_requests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inference-lab-test-") as directory:
            temp = Path(directory)
            manifest, metadata = temp / "prompts.json", temp / "metadata.json"
            manifest.write_text('["synthetic"]', encoding="utf-8")
            data = fixture("test")["config"]["metadata"]
            data["model_revision"] = "TODO: record real revision"
            metadata.write_text(json.dumps(data), encoding="utf-8")
            args = ["client.py", "--manifest", str(manifest), "--metadata", str(metadata),
                    "--label", "test", "--output", str(temp / "out.json")]
            with patch.object(sys, "argv", args), patch.object(client, "benchmark") as runner:
                with self.assertRaisesRegex(ValueError, "model_revision"):
                    client.main()
                runner.assert_not_called()


class ComparisonTests(unittest.TestCase):
    def test_matched_pair(self) -> None:
        result = compare(fixture("left"), fixture("right"))
        self.assertEqual(result["metrics"]["output_tokens_per_s"], {"left": 10, "right": 10})

    def test_config_mismatches(self) -> None:
        for field in ("model", "metadata", "manifest_sha256", "max_tokens",
                      "concurrency", "requests", "warmup", "temperature", "endpoint"):
            with self.subTest(field=field):
                right = fixture("right")
                right["config"][field] = "different"
                with self.assertRaisesRegex(ValueError, field):
                    validate_pair(fixture("left"), right)

    def test_actual_lengths_and_order_must_match(self) -> None:
        for field in ("prompt_index", "prompt_tokens", "output_tokens"):
            right = fixture("right")
            right["rows"][0][field] += 1
            with self.assertRaisesRegex(ValueError, "Actual"):
                validate_pair(fixture("left"), right)
        with self.assertRaises(ValueError):
            validate_pair(fixture("same"), fixture("same"))

    def test_prefix_conditions(self) -> None:
        workloads = prefix_workloads(32)
        self.assertEqual(len(set(workloads["shared"])), 32)
        self.assertEqual(workloads["shared"][0][:100], workloads["shared"][1][:100])
        self.assertNotEqual(workloads["divergent"][0][:100], workloads["divergent"][1][:100])


class WorkspaceTests(unittest.TestCase):
    def test_cuda_unavailable_is_explicit_skip(self) -> None:
        output = io.StringIO()
        with patch("torch.cuda.is_available", return_value=False), redirect_stdout(output):
            self.assertFalse(require_cuda())
        self.assertIn("SKIP", output.getvalue())
        self.assertIn("No CPU timing", output.getvalue())

    def test_new_days_are_complete_and_knowledge_is_substantial(self) -> None:
        folders = sorted(day for week in ROOT.glob("week_0[5-7]*") for day in week.glob("day_*"))
        self.assertEqual(len(folders), 21)
        for folder in folders:
            with self.subTest(folder=folder.name):
                text = (folder / "README.md").read_text(encoding="utf-8")
                for heading in ("**Goal:**", "## Knowledge", "## References", "**Done when:**"):
                    self.assertIn(heading, text)
                knowledge = text.split("## Knowledge\n", 1)[1].split("\n## ", 1)[0]
                self.assertGreaterEqual(len(knowledge.split()), 150)
                self.assertLessEqual(len(knowledge.split()), 400)
                if (folder / "exercise.py").exists():
                    self.assertTrue((folder / "solution.py").exists())
                else:
                    self.assertTrue((folder / "worksheet.md").exists())

    def test_python_syntax(self) -> None:
        for folder in ROOT.glob("week_0[5-7]*"):
            for source in folder.rglob("*.py"):
                ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def test_local_markdown_links(self) -> None:
        files = [ROOT / name for name in ("README.md", "NOTE.md", "inference_engineer_roadmap.md")]
        files += [file for folder in ROOT.glob("week_0[5-7]*") for file in folder.rglob("*.md")]
        for file in files:
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(file=file.name, target=target):
                    self.assertTrue((file.parent / target.split("#")[0]).exists(), target)


if __name__ == "__main__":
    unittest.main()
