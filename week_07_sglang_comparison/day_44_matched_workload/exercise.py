"""Verify actual token accounting, not only a shared max_tokens request."""


def token_lengths_match(left: list[dict], right: list[dict]) -> bool:
    # TODO: match request counts, prompt index order, prompt tokens and output tokens.
    raise NotImplementedError("Detect unequal work before comparing timing.")


if __name__ == "__main__":
    rows = [{"prompt_index": 0, "prompt_tokens": 12, "output_tokens": 8}]
    assert token_lengths_match(rows, rows)
    assert not token_lengths_match(rows, [])
    assert not token_lengths_match(rows, [{**rows[0], "output_tokens": 7}])
