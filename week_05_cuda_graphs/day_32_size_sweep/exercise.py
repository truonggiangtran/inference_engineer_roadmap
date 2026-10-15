"""Compute a speedup from matching timing boundaries, not wall/event mixtures."""


def speedup(eager_wall_ms: float, replay_wall_ms: float) -> float:
    # TODO: validate positive measurements, then return eager / replay.
    raise NotImplementedError("Compute a same-boundary speedup.")


if __name__ == "__main__":
    assert speedup(4.0, 2.0) == 2.0
    assert speedup(2.0, 4.0) == 0.5
    try:
        speedup(1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero duration must be rejected.")
    print("Checks passed. Implement and run the size sweep described in README.md.")
