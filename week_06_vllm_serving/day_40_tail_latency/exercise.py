"""Implement linearly interpolated sample quantiles, retaining the raw observations."""
import math


def percentile(values: list[float], q: float) -> float:
    # TODO: sort finite values, validate q, interpolate index (n - 1) * q.
    raise NotImplementedError("Implement a descriptive sample quantile.")


if __name__ == "__main__":
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert math.isclose(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)
    assert percentile([7.0], 0.95) == 7.0
    for values, q in (([], 0.5), ([1.0], 1.1), ([float("nan")], 0.5)):
        try:
            percentile(values, q)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid quantile input accepted.")
