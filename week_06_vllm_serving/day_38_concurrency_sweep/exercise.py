"""Build stable synthetic inputs; do not regenerate random prompts per framework."""


def make_prompts(count: int) -> list[str]:
    # TODO: create count unique deterministic prompts with early varying identifiers.
    raise NotImplementedError("Build the manifest, then run the concurrency sweep.")


if __name__ == "__main__":
    prompts = make_prompts(32)
    assert len(prompts) == len(set(prompts)) == 32
    assert prompts == make_prompts(32)
    assert all(isinstance(prompt, str) and prompt for prompt in prompts)
