"""Reject an apples-to-oranges comparison before calculating any speed ratio."""
from copy import deepcopy


def validate_pair(left: dict, right: dict) -> None:
    # TODO: compare manifest, model/revisions/dtype/hardware, settings and actual token lengths.
    # Raise ValueError naming the differing field. See the reference input schema.
    raise NotImplementedError("Implement the compatibility gate before performance comparison.")


if __name__ == "__main__":
    left = {
        "config": {
            "model": "toy", "metadata": {"dtype": "float32"},
            "manifest_sha256": "toy-test-only", "max_tokens": 8,
            "concurrency": 1, "requests": 1, "warmup": 0, "temperature": 0,
            "endpoint": "raw text /v1/completions", "label": "left",
        },
        "rows": [{"prompt_index": 0, "prompt_tokens": 12, "output_tokens": 8}],
    }
    right = deepcopy(left)
    right["config"]["label"] = "right"
    validate_pair(left, right)
    right["rows"][0]["output_tokens"] = 7
    try:
        validate_pair(left, right)
    except ValueError:
        print("Mismatch correctly rejected; now test your actual measurements.")
    else:
        raise AssertionError("Unequal generated lengths must be rejected.")
