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

## Model catalog

[`python/src/ol_llm/catalog.toml`](python/src/ol_llm/catalog.toml) records the models
applications call, with each one's status, whether the id floats, the name Opik prices it
under, and its list price. It is not an allowlist, so a new model needs no release here:
`build_model` builds an id the catalog does not list, logs a warning, and tags the run's trace
`model_status:uncataloged`. It refuses a retired id, and tags a deprecated one
`model_status:deprecated`. A model with no entry has no price in the spend rules, so a model
that stays in use should get one. Adding or changing an entry is a pull request here; the
policy is
[`llm-usage-policies-spec.md` §2.2](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-usage-policies-spec.md).

An application keeps its own model lists (database rows, settings) to the catalog in CI, so
a release that retires a model fails the CI of any app still naming it. Deprecated and
unlisted ids are warnings:

```sh
ol-llm-catalog check openai:gpt-4o-mini azure:gpt-5-mini
ol-llm-catalog check --file models.txt   # one id per line, # comments
```

or from a test, `assert ol_llm.check_model_ids(ids).ok`. `ol-llm-catalog prices` prints the
prices as a Prometheus recording rule group for the spend dashboard and alerts.

The test suite checks that genai-prices can price every chat model, which a `cost_limit`
needs, and that Opik prices every model at the catalog price. The Opik check downloads
Opik's price table, so offline runs deselect it with `-m "not network"`.

## Development

```sh
uv sync
uv run prek install
uv run pytest python/tests
```

## Releasing

Bump `version` in `python/pyproject.toml` in a pull request; `ol_llm.__version__` reads it
from the installed metadata. When it merges, the publish workflow builds,
publishes to PyPI with Trusted Publishing, and tags the commit `ol-llm-v<version>`. A merge
that does not change the version publishes nothing.
