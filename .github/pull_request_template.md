## What and why

<!-- One or two sentences: the problem this change solves. Link the issue if there is one. -->

## How

<!-- The approach, and alternatives you rejected if they are not obvious. -->

## Evaluation impact

<!-- Retrieval or answer quality changes? Paste the before/after rows from `ragtr eval`. If the `ci` suite
     changed on purpose, regenerate eval/results/ci.json in this PR and explain why. -->

## Checklist

- [ ] `uv run ruff check`, `uv run ruff format --check`, `uv run mypy` and `uv run pytest` pass (Docker running for integration tests)
- [ ] Tests cover the new behaviour, including failure paths
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] A new ADR in `docs/adr/` if this changes an architectural decision
- [ ] No secrets, real personal data or customer data
