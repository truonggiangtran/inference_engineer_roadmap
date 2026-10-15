"""Build two prefix locality conditions; verify actual token lengths separately."""


def prefix_workloads(count: int) -> dict[str, list[str]]:
    # TODO: long common start in 'shared', early unique IDs in 'divergent'.
    raise NotImplementedError("Create explicit prefix-locality conditions.")


if __name__ == "__main__":
    data = prefix_workloads(32)
    assert set(data) == {"shared", "divergent"}
    assert all(len(rows) == len(set(rows)) == 32 for rows in data.values())
    assert data["shared"][0][:100] == data["shared"][1][:100]
    assert data["divergent"][0][:100] != data["divergent"][1][:100]
