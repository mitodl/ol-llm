# ol-llm

Shared library for LLM calls at MIT Open Learning, with Opik tracing, prompt management, and
evaluation built in.

| Path | What it holds |
| --- | --- |
| `python/` | The `ol-llm` distribution on PyPI (import `ol_llm`) |
| `typescript/` | Reserved for a TypeScript client; empty until a production JS/TS call site exists |
| `docs/` | Language-neutral contracts both clients implement |

The design and the decisions behind it are in
[mitodl/ol-infrastructure `docs/plans/shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md).
The contracts move into `docs/` here as they are implemented.

## Development

```sh
uv sync
uv run prek install
uv run pytest python/tests
```

## Releasing

Bump `version` in `python/pyproject.toml` and `__version__` in
`python/src/ol_llm/__init__.py` in a pull request. When it merges, the publish workflow builds,
publishes to PyPI with Trusted Publishing, and tags the commit `ol-llm-v<version>`. A merge
that does not change the version publishes nothing.
