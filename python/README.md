# ol-llm

The library every LLM call site at MIT Open Learning goes through. It owns the Opik
integration (tracing, prompt management, evaluation), a single timeout and retry policy, and
the attribution fields that let cost, quality, and traffic join across applications.

This release is a placeholder. It reserves the name, proves the release pipeline, and does one
thing: importing `ol_llm` sets `OPIK_SENTRY_ENABLE=false` before anything imports the Opik SDK,
which otherwise replaces the host application's Sentry client with its own.

The design is in [mitodl/ol-infrastructure `docs/plans/shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md).

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
