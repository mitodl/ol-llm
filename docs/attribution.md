# Attribution schema

The fields every LLM call carries, so cost, quality, and traffic join across Opik traces,
gateway logs, and metrics. Decided 2026-10-01; the reasoning is in
[`shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md)
§4. The library applies the fields; call sites supply the values.

| Field | When | Source | On the Opik trace | Elsewhere |
| --- | --- | --- | --- | --- |
| `app` | always | the use case | project name, tag `app:<app>`, metadata | gateway consumer |
| `use_case` | always | the use case | tag `use_case:<name>`, metadata | `X-OL-LLM-Use-Case` header on gateway requests |
| `environment` | always | deployment configuration | native environment field, tag `env:<environment>`, metadata | |
| `model`, `provider` | always | the model that answered | `gen_ai.request.model`, `gen_ai.provider.name` on spans | |
| `prompt_name`, `prompt_version` | a managed prompt is used | the prompt helper | the native prompt link, plus metadata | |
| `user_id` | request-time with a user | the call | metadata | OpenAI `user`, Anthropic `metadata.user_id` |
| `thread_id` | conversations | the call | native thread field | |
| `run_id` | batch | the call | metadata | |
| `model_status` | a model in the use case's chain is deprecated or not in the catalog | the model catalog | tag `model_status:deprecated` or `model_status:uncataloged` | |
| `degraded` | when true | the library, or the call site for partial output | `degraded`, `degraded_reason` metadata | |

Tags carry only `app`, `use_case`, `environment`, and `model_status`. User, thread, and run
identifiers go in metadata and never in tags or metric labels.

`user_id` is pseudonymous and stable: the platform's global user id for a signed-in user, and
for an anonymous one `anon-` followed by an HMAC-SHA256 of the session identifier under one
platform-wide key. Emails, usernames, and raw session keys never leave the application.

## Configuration

`environment` comes from `OL_LLM_ENVIRONMENT`, or from a `Settings` object the host passes to
`configure` at startup (Django settings, a Dagster resource). Gateway routes are configured
the same way, per API shape:

| Variable | Meaning |
| --- | --- |
| `OL_LLM_ENVIRONMENT` | required; e.g. `ci`, `qa`, `production` |
| `OL_LLM_GATEWAY_OPENAI_BASE_URL`, `OL_LLM_GATEWAY_OPENAI_API_KEY` | send `openai`, `openai-chat`, and `openai-responses` models through this OpenAI-format route |
| `OL_LLM_GATEWAY_ANTHROPIC_BASE_URL`, `OL_LLM_GATEWAY_ANTHROPIC_API_KEY` | send `anthropic` models through this Anthropic-format route |

A family with no route is called directly with the provider SDK's own credentials. The gateway
is optional in phase one, and which routes exist is the gateway PoCs' to decide. The model
catalog records whether each model it lists is reached through a gateway route, but the
routing itself comes from these settings.
