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

Run the agent with `ol_llm.run`, or `ol_llm.run_sync` in a Celery task or Dagster asset:

```python
result = ol_llm.run_sync(
    summarizer,
    text,
    attribution=ol_llm.Attribution(user_id=user.global_id, run_id=task_id),
)
result.output  # a Summary
result.degraded  # True when a fallback model answered
```

Each run is one Opik trace in the app's project, tagged with the use case and environment,
with the model calls, token usage, and cost inside it. Model failures raise `LLMError`
subclasses, and invalid output after its retries raises `LLMInvalidOutput`. A run inside the
application's own `@opik.track` function becomes a span of that function's trace.
`run_stream` and the single-call `complete` are not released yet.

### Tracing

Tracing reads the variables the Opik SDK reads, so the SDK and the span exporter always send
to the same instance:

| Variable | Meaning |
| --- | --- |
| `OPIK_URL_OVERRIDE` | the Opik API, e.g. `https://opik-qa.ol.mit.edu/api` |
| `OPIK_WORKSPACE` | `default` on our instances |
| `OPIK_KEYCLOAK_TOKEN_URL`, `OPIK_KEYCLOAK_CLIENT_ID`, `OPIK_KEYCLOAK_CLIENT_SECRET` | the Keycloak client the app authenticates to Opik with; `OPIK_KEYCLOAK_SCOPE` if it needs one. Leave `OPIK_API_KEY` unset |
| `OPIK_TRACK_DISABLE` | `true` turns tracing off, for local development without an Opik |
| `OL_LLM_TRACE_CONTENT` | `false` keeps prompts, completions, and tool arguments out of traces. On by default until the data-handling policy decides |

Call `ol_llm.configure_tracing()` at startup (Django's `AppConfig.ready`, a Dagster resource)
before anything else touches the Opik SDK: the SDK reads the Keycloak hook when it first
builds its HTTP client. Short-lived processes should call `ol_llm.tracing.flush()` before
they exit. An app that registers its own Keycloak hook on the SDK (learn-ai's
`opik_keycloak_auth`) should drop it when it adopts the library.

### Testing

`ol_llm.testing.override_models` makes `build_model` and `build_agent` use PydanticAI's
`TestModel` or `FunctionModel` in place of a provider, still under the library's retry and
error handling. Set `OPIK_TRACK_DISABLE=true` for a test suite that should send nothing, or
use `ol_llm.testing.tracing_disabled` around one test. `opik.record_traces_locally()`
captures the root traces a test produces. Script responses and failures with
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
