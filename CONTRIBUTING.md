# Contributing

Thanks for taking the time to improve rag-tr.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.9 or later (it installs Python 3.13 from `.python-version` if needed)
- Docker (integration tests start pgvector/PostgreSQL through Testcontainers; `docker compose up` runs the demo)

## Workflow

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Branch from `main` and keep each pull request to one concern.
3. Before pushing:

   ```bash
   uv sync                      # dev tools; add --extra ml for real embedders and rerankers
   uv run ruff check && uv run ruff format --check
   uv run mypy                  # strict
   uv run pytest                # unit + Testcontainers integration + the evaluation regression gate
   ```

## Changes that move retrieval or answer quality

`tests/unit/test_harness.py` re-runs the download-free `ci` evaluation suite and fails when any metric
differs from `eval/results/ci.json`. That is deliberate: a change to tokenisation, chunking, fusion or the
extractive generator must show its effect. If the change is intended:

```bash
uv run ragtr eval --suite ci --embedder hashing-384      # rewrites eval/results/ci.{json,md}
```

Commit the new result in the same pull request and explain the movement in the description. For changes
that affect the model-backed numbers, re-run the relevant suite (`main`, `chunking`, `normalisation`,
`embedders`, `generation`) and update `docs/evaluation.md`.

The data set is generated, not hand-edited: change `src/ragtr/data/synth.py` and run
`uv run ragtr generate-data`. A unit test checks that `data/` is byte-identical to the generator output.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) are required. They keep the history
readable and make the hand-written CHANGELOG easy to assemble at release time:

- `feat:` new capability (minor bump)
- `fix:` bug fix (patch bump)
- `feat!:` or a `BREAKING CHANGE:` footer (major bump)
- `docs:`, `test:`, `perf:`, `refactor:`, `build:`, `ci:`, `chore:` for everything else

## Dependencies

There is no dependency bot. Versions are locked in `uv.lock` and updated by hand (`uv lock --upgrade-package
<name>`), with the reason in the commit message. CI fails on any package with a known vulnerability.
