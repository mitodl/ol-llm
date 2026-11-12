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
- `opik` is bumped once a month (the Renovate rule in `renovate.json` opens the PR on the 1st),
  so the version caps it puts on its dependencies stay close to current. On a bump, list every
  requirement of the new release with an upper bound and review them by hand (caps are mostly
  `<=`, but opik also bounds litellm with `<1.106`):
  `curl -s https://pypi.org/pypi/opik/<version>/json | jq -r '.info.requires_dist[] | select(test("<"))'`.
  opik states a capped package's floor or extra on a separate line (e.g.
  `boto3-stubs[bedrock-runtime]>=1.34.110` beside `boto3-stubs<=1.43.123`), so read all of
  its requirements on each capped package, not only the `<` lines.
  Replace the block in the README's "Overriding opik's version caps" section with one entry per
  capped package. Write each entry as all the non-cap bounds and extras of every package that
  requires it, opik included, combined (list them with
  `uv tree --invert --package <name> --depth 1 --no-dev`, then read each one's requirement on
  it), with opik's environment markers kept, e.g. `"pydantic>=2.14,<3"`,
  `"tenacity>=8.2.3,!=8.4.0,<9.2.0"` (langchain-core excludes 8.4.0, google-genai caps 9.2).
  The litellm entry is the exception: it replaces opik's `<1.106` bound with `>=1.108`, the
  first litellm that accepts the `openai` extra's openai 3, and the root `pyproject.toml`
  overrides litellm the same way. `python/smoke/pyproject.toml` takes the cap entries for the
  packages its slimming list does not drop. Keep the smoke test passing with them. Drop them
  once comet-ml/opik#9491 moves the caps into an extra.
- Do not add litellm as a dependency.
- No async client is created outside a `LoopLocal` (`ol_llm/loops.py`) with a close function,
  so the runner can close it and a forked child drops it. New loop-local resources join the
  lifecycle matrix in `python/tests/test_loops.py`.
- Sync entry points run their coroutine with `run_coroutine`, never `asyncio.run` or a
  PydanticAI `*_sync` method, and the README tells apps to do the same.
- A change a consumer would notice adds a fragment with `uv run scriv create --edit`
  (`changelog.d/README.md`). Never edit `CHANGELOG.md` directly outside a release.
- A release bumps the version in `python/pyproject.toml`, the only place the version lives,
  and runs `uv run scriv collect`; the publish workflow does the rest on merge.
