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
`build_model` and `embed` build an id the catalog does not list, log a warning, and tag the
trace `model_status:uncataloged`. They refuse a retired id, and tag a deprecated one
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
needs, and every embedding model, and that Opik prices every model at the catalog price. The
Opik check downloads Opik's price table, so offline runs deselect it with `-m "not network"`.

## Decision models

A decision ("system-one") model answers typed questions about a text in one forward pass,
with probabilities and no generated text, in tens of milliseconds. Use one where an app
would otherwise match phrase lists or ask a chat model for JSON: routing, intent, yes/no
checks. They are PydanticAI decision models, so a use case names one like any other model
and `complete` (or `run`) runs it with the `pydantic-ai` extra. Each field of the output
type is one question: a `bool` is a yes/no, a `Literal` or `Enum` of strings a pick-one,
and its description is the question. The confidence and probabilities come back in
`LLMResult.provider_details`:

```python
class ReplyIntent(BaseModel):
    """What the learner wants done with the support request we showed them."""

    intent: Literal["send", "cancel", "edit"] = Field(
        description="What is the learner telling us to do with the request?"
    )


REPLY_INTENT = ol_llm.UseCase(
    app="learn-ai",
    name="support-reply-intent",
    profile="request",
    model="azure-decision:Microsoft-Decision-1.1",
)

result = await ol_llm.complete(REPLY_INTENT, "yeah go for it", output_type=ReplyIntent)
probability = result.provider_details["probabilities"]["intent"][result.output.intent]
```

The model id's provider picks the server:

| Provider | Server | Configuration |
| --- | --- | --- |
| `azure-decision:<deployment>` | Microsoft-Decision on an Azure Foundry (AIServices) account | `OL_LLM_AZURE_DECISION_ENDPOINT`, and `Settings.azure_ad_token_provider` for the Entra token |
| `system-one:<model>` | Any System One server: DigitalOcean, OVHcloud, Telnyx, Cloudflare's Clef through one of them, or a self-hosted Laya (`laya-serve`), Strands Decider, Ollama or `llama-server` | `OL_LLM_SYSTEMONE_BASE_URL`, and `OL_LLM_SYSTEMONE_API_KEY` if the server checks one |
| `typesafe:<model>` | TypeSafe's hosted Jev | `TYPESAFE_API_KEY` |
| `openai-decisions:<model>` | OpenAI's Decisions API, e.g. `gpt-6-luna` | The `openai` extra, `OPENAI_API_KEY` or the `openai` gateway route |
| `google-decisions:<model>` | Gemini's Decisions API, e.g. `gemini-decision-1` | The `google` extra and `GOOGLE_API_KEY` |

Calls run under the use case's policy like a chat model's: its timeout, retries and error
classes, its trace, and its catalog checks. A decision use case can name a chat model in
`fallback_models`, which takes over on an outage. Its answer carries no decision
probabilities, so check for `"fallback_model"` in `result.degraded_reasons` before reading
`provider_details`. A decision model that hands a step off
(PydanticAI's `UnsureRoute` or `UnfillableRoute`) is not retried and reaches the caller as
itself. Thresholds are per model, since each calibrates its probabilities differently, and
an app should keep a deterministic path for when the call fails or is not sure.

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
