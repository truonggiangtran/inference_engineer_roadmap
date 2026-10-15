"""Calculate observed chunk gaps, without pretending chunks are tokens."""


def chunk_gaps(arrivals_ms: list[float]) -> list[float]:
    # TODO: require nonnegative ordered arrivals; return adjacent differences.
    raise NotImplementedError("Implement chunk-gap observations, then run the streaming lab.")


if __name__ == "__main__":
    assert chunk_gaps([10.0, 12.0, 17.0]) == [2.0, 5.0]
    assert chunk_gaps([10.0]) == []
    try:
        chunk_gaps([2.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-order arrivals must be rejected.")
