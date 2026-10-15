"""Select a supported shape rather than silently truncating an oversized input."""


def choose_bucket(length: int, buckets: tuple[int, ...]) -> int:
    # TODO: return the smallest fitting bucket; reject nonpositive/oversized inputs.
    raise NotImplementedError("Implement explicit shape routing.")


if __name__ == "__main__":
    assert choose_bucket(200, (4096, 256, 1024)) == 256
    assert choose_bucket(1024, (256, 1024)) == 1024
    for length in (0, -1, 4097):
        try:
            choose_bucket(length, (256, 1024, 4096))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid length accepted: {length}")
    print("Routing checks passed. Add retained graph cases as described in README.md.")
