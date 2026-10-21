# ol-llm: agent guide

The shared library every MIT Open Learning LLM call site goes through. The design is in
mitodl/ol-infrastructure `docs/plans/shared-llm-library-spec.md`. Read §3 (error contract), §4
(attribution), and §6 (call contract) before changing behavior; the code implements them.

## Layout

- `python/`: the `ol-llm` distribution. Source in `python/src/ol_llm`, tests in `python/tests`.
- `python/smoke/`: a separate uv project with its own lockfile that installs ol-llm the way a
  slimmed consumer does. It is excluded from the workspace on purpose.
- `typescript/`: reserved. `docs/`: language-neutral contracts.

## Commands

```sh
uv sync                                         # workspace, all provider extras, dev tools
uv run pytest python/tests
uv run mypy python/src
uv run prek run --all-files                     # the same hooks CI runs
(cd python/smoke && uv sync && uv run python smoke_test.py)
```

## Rules that are easy to break

- `ol_llm/__init__.py` sets `OPIK_SENTRY_ENABLE=false` and `OPIK_ANALYTICS_ENABLE=false` before
  anything imports opik. Nothing may import opik ahead of it.
- `opik` is pinned exactly. Bumping it means the smoke test must still pass with the overrides
  in `python/smoke/pyproject.toml`, which are the overrides consumers copy from the README.
- Do not add litellm as a dependency.
- A release is a version bump in `python/pyproject.toml`, the only place the version lives;
  the publish workflow does the rest on merge.
