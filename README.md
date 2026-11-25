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
`build_model`, `embed` and `decide` build an id the catalog does not list, log a warning, and tag the
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

A decision ("system-one") model answers typed questions about a state in one forward pass,
with probabilities and no generated text, in tens of milliseconds. Use one where an app
would otherwise match phrase lists or ask a chat model for JSON: routing, intent, yes/no
checks. `ol_llm.decide` (and `decide_sync`) sends the System One API request that
Microsoft-Decision, TypeSafe's Jev, Laya and Strands Decider all accept:

```python
REPLY_INTENT = ol_llm.UseCase(
    app="learn-ai",
    name="support-reply-intent",
    profile="request",
    model="azure:Microsoft-Decision-1.1",
)

decision = await ol_llm.decide(
    REPLY_INTENT,
    {"we_asked": "Send this request to support?", "reply": "yeah go for it"},
    {
        "intent": ol_llm.Choice(
            instructions="What is the learner telling us to do with the request?",
            options={
                "send": None,
                "cancel": None,
                "edit": "They added or changed details",
            },
        ),
    },
)
intent = decision.choice("intent")
if intent.confidence is not None and intent.confidence >= 0.8:
    ...
```

The model id's provider picks the server:

| Provider | Server | Configuration |
| --- | --- | --- |
| `azure:<deployment>` | Microsoft-Decision on an Azure Foundry (AIServices) account | `OL_LLM_AZURE_DECISION_ENDPOINT`, and `Settings.azure_ad_token_provider` for the Entra token |
| `typesafe:<model>` | TypeSafe's hosted Jev | `TYPESAFE_API_KEY` |
| `systemone:<model>` | A self-hosted Laya (`laya-serve`) or Strands Decider (`strands-decider serve`) | `OL_LLM_SYSTEMONE_BASE_URL`, and `OL_LLM_SYSTEMONE_API_KEY` if the server checks one |

`ChoiceAnswer.confidence` is computed by ol-llm from the chosen option's probability, the
same way for every model, because each vendor's own `confidence` uses a different formula.
`decide` makes one request with no output to bound, so a use case's `request_limit` and
`cost_limit` don't apply to it. Thresholds are
still per model: each one calibrates its probabilities differently, which is also why a
decision use case takes no `fallback_models`. Keep a deterministic path for when the call
fails or is not confident. Tests can answer requests with
`ol_llm.decisions.override_decision_transport(httpx2.MockTransport(handler))`.

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
