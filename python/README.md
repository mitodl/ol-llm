# ol-llm

The library every LLM call site at MIT Open Learning goes through. It owns the Opik
integration (tracing, prompt management, evaluation), a single timeout and retry policy, and
the attribution fields that let cost, quality, and traffic join across applications.

Importing `ol_llm` sets `OPIK_SENTRY_ENABLE=false` before anything imports the Opik SDK, which
otherwise replaces the host application's Sentry client with its own, so import it first.

The design is in [mitodl/ol-infrastructure `docs/plans/shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md).
The rules the code follows are in [`docs/`](https://github.com/mitodl/ol-llm/tree/main/docs):
the [error contract](https://github.com/mitodl/ol-llm/blob/main/docs/error-contract.md) and the
[attribution schema](https://github.com/mitodl/ol-llm/blob/main/docs/attribution.md).

## Usage

Declare each call site once, as a module-level constant, and build its agent from it:

```python
from pydantic import BaseModel

import ol_llm

SUMMARY = ol_llm.UseCase(
    app="mit-learn",
    name="content-summary",
    profile="batch",
    model="openai:gpt-4o-mini",
    fallback_models=("bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0",),
)


class Summary(BaseModel):
    text: str


summarizer = ol_llm.build_agent(SUMMARY, output_type=Summary, instructions="Summarize.")
```

The agent's model has a timeout on every request, retries rate limits and outages and
nothing else, falls back once the primary has spent its retries, and raises `LLMError`
subclasses (`LLMRateLimited`, `LLMUnavailable`, `LLMRejected`, ...) with the provider's
exception as the cause. Build agents once, not per request: each build creates new provider
clients.

Set `OL_LLM_ENVIRONMENT`, or call `ol_llm.configure(ol_llm.Settings(environment=...))` at
startup. Provider credentials are each SDK's own (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, the AWS
and Google default credential chains, and so on). Operators can change a use case's limits
without a release, e.g. `OL_LLM_CONTENT_SUMMARY_TIMEOUT_SECONDS=120`.

Call sites not built on PydanticAI (LangGraph, a plain SDK) apply the same numbers with
`ol_llm.resolve_policy(use_case)`, which needs no extra.

The entry points that wrap a run in an Opik trace (`run`, `run_stream`, `complete`) are not
released yet; until they are, `build_agent` gives the policy and errors without the tracing.

### Testing

`ol_llm.testing.override_models` makes `build_model` and `build_agent` use PydanticAI's
`TestModel` or `FunctionModel` in place of a provider, still under the library's retry and
error handling, and `tracing_disabled` turns Opik off. Script responses and failures with
`FunctionModel`; do not commit recorded provider responses.

## Extras

The base install is the Opik SDK and OpenTelemetry, which every call site needs. Everything
else is opt-in:

| Extra | Installs | For |
| --- | --- | --- |
| `pydantic-ai` | `pydantic-ai-slim` | The default construction path for agents and single calls |
| `openai`, `anthropic`, `bedrock`, `google`, `mistral` | the matching `pydantic-ai-slim` extra | A provider SDK; each also installs PydanticAI |
| `harness` | `pydantic-ai-harness` | In-session memory, compaction, and conversation search. Each harness release pins PydanticAI exactly |
| `memory` | `hindsight-client` | The shared memory service's client. Provisional: the backend is still being evaluated |

For example, `ol-llm[bedrock,harness]`.

## Installing without the Opik SDK's heavy dependencies

The `opik` package hard-requires litellm, openai, pytest, and several other packages the
tracing and prompt paths never import. uv consumers can drop them:

```toml
[tool.uv]
override-dependencies = [
    "boto3-stubs; sys_platform == 'never'",
    "click; sys_platform == 'never'",
    "litellm; sys_platform == 'never'",
    "openai; sys_platform == 'never'",
    "pytest; sys_platform == 'never'",
    "rapidfuzz; sys_platform == 'never'",
    "tree-sitter; sys_platform == 'never'",
    "tree-sitter-javascript; sys_platform == 'never'",
    "tree-sitter-typescript; sys_platform == 'never'",
    "watchfiles; sys_platform == 'never'",
]
```

Drop `openai` from that list if the application uses the `openai` extra. Opik's evaluation
metrics need litellm, so the overrides belong in runtime images, not in an environment that
runs evaluations. CI checks this exact list on every change (`python/smoke/`).

Applications that cap how new a dependency may be (`exclude-newer` in uv) should exempt this
package so releases are usable the day they ship:

```toml
[tool.uv.exclude-newer-package]
ol-llm = "0d"
```
