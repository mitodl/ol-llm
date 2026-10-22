# ol-llm

The library every LLM call site at MIT Open Learning goes through. It owns the Opik
integration (tracing, prompt management, evaluation), a single timeout and retry policy, and
the attribution fields that let cost, quality, and traffic join across applications.

Importing `ol_llm` sets `OPIK_SENTRY_ENABLE=false` before anything imports the Opik SDK, which
otherwise replaces the host application's Sentry client with its own, so import it first. It
also sets `OPIK_ANALYTICS_ENABLE=false`, which stops the SDK reporting which of its features
get used to Comet. Either one set explicitly in the environment is left alone.

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
clients. An agent built once can run on any thread or event loop (`run_sync` from a thread pool,
`asyncio.run` per task); its model keeps one set of provider clients per event loop, released
with the thread that ran the loop. A call site that needs a new agent per run (per-run
instructions, output schema, or validators) builds the model once with `ol_llm.build_model` and
each agent over it with `ol_llm.agent_over(model, ...)`, which takes the arguments
`build_agent` does after the use case. Agents over one model share its provider clients.

Set `OL_LLM_ENVIRONMENT`, or call `ol_llm.configure(ol_llm.Settings(environment=...))` at
startup. Provider credentials are each SDK's own (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, the AWS
and Google default credential chains, and so on). Operators can change a use case's limits
without a release, e.g. `OL_LLM_CONTENT_SUMMARY_TIMEOUT_SECONDS=120`.

Call sites not built on PydanticAI (LangGraph, a plain SDK) apply the same numbers with
`ol_llm.resolve_policy(use_case)`, which needs no extra. Their embedding requests go through
`ol_llm.traced_embedding` (see Embeddings).

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
subclasses, and invalid output after its retries raises `LLMInvalidOutput`.

A run inside the application's own `@opik.track` function becomes a span of that function's
trace, in that function's project: Opik files a span under its trace's project, and the model
spans follow. The use case's tags go on the outer trace so tag filters still find it; the
attribution metadata stays on the run's span.

A single model call needs no agent. `ol_llm.complete` (or `ol_llm.complete_sync`) takes the use
case and the prompt, or the conversation so far as a list of PydanticAI messages ending in the
request to send:

```python
from pydantic_ai import ToolOutput

# Plain text.
result = await ol_llm.complete(SUMMARY, text, instructions="Summarize in one sentence.")
result.output  # a str

# Structured output, validated into the schema and retried on the same model when invalid.
result = ol_llm.complete_sync(
    SUMMARY, text, output_type=Summary, instructions="Summarize."
)
result.output  # a Summary


# A forced tool call: the model gets one tool and no text output, so it has to call it.
class Rubric(BaseModel):
    score: int
    rationale: str


result = ol_llm.complete_sync(
    GRADING,
    submission,
    output_type=ToolOutput(
        Rubric, name="record_grade", description="Record the grade."
    ),
    attribution=ol_llm.Attribution(user_id=user.global_id),
)
result.output  # a Rubric
```

`model_settings` apply to every request, e.g. `max_tokens`. The timeout and the provider user id
fields are the library's, and setting them raises `ValueError`. PydanticAI leaves a
`ToolOutput`'s tool unforced on a request that would think, since forcing a tool turns thinking
off (Claude models that think by default); naming the tool in `tool_choice` forces it.
`output_validators` run on the output once it matches the type, and one that raises PydanticAI's
`ModelRetry` sends the output back to the model with its message, within the use case's
`output_retries`:

```python
from pydantic_ai import ModelRetry


def score_in_range(rubric: Rubric) -> Rubric:
    if not 0 <= rubric.score <= 10:
        raise ModelRetry("score must be between 0 and 10")
    return rubric


result = ol_llm.complete_sync(
    GRADING,
    submission,
    output_type=ToolOutput(Rubric, name="record_grade"),
    model_settings={"max_tokens": 4000, "tool_choice": ["record_grade"]},
    output_validators=[score_in_range],
)
```

`build_agent` and `agent_over` take both too. An agent with tools may not force a tool, which
would make every request call one.

`complete` runs a tool-less agent over the use case's model through `run`, so the timeout,
retries, fallback, trace, errors, and `LLMResult` are the same. The model is built on the first
call for a use case and reused after that. A call site that grows tools or more steps moves to
the agent path by building the same agent itself, with the arguments it already passes:

```python
grader = ol_llm.build_agent(
    GRADING, output_type=ToolOutput(Rubric, name="record_grade"), tools=[lookup_rubric]
)
result = ol_llm.run_sync(grader, submission, attribution=attribution)
```

Request-time call sites that show the answer as it arrives use `ol_llm.run_stream`, an async
context manager that takes the same arguments as `run`:

```python
async with ol_llm.run_stream(tutor, message, attribution=attribution) as stream:
    async for text in stream.stream_text():
        await send(text)
    result = await stream.result()
```

`stream_text()` yields each new piece of text (`delta=False` yields the whole text so far), and
`stream_output()` yields structured output as it validates. `result()` returns the same
`LLMResult` as `run`, and runs `postprocess`; call it inside the block, since the trace closes
when the block ends. A block that reads the whole stream without calling `result()` still marks
the trace. The run's trace stays current for the whole block, so the model spans land in it. A
failure before the first chunk is retried and falls back as in `run` and raises on entry; a
failure after it raises `LLMUnavailable` (or another `LLMError`) where the stream is read, and
is not retried.

### Prompts

Prompts live in the app's Opik project and are fetched by environment label. Each call site
passes an embedded default: the text of the version production points at, exported into the
image at build time.

```python
SUMMARY_PROMPT = ol_llm.TextTemplate(template=EXPORTED_TEXT, version="v7")


def summarize(text: str, audience: str) -> Summary:
    prompt = ol_llm.get_prompt(
        "content-summary", project_name="mit-learn", default=SUMMARY_PROMPT
    )
    result = ol_llm.complete_sync(
        SUMMARY,
        text,
        output_type=Summary,
        instructions=prompt.format(audience=audience),
        prompt=prompt,
    )
    return result.output  # result.prompt_version is "v7"
```

The label defaults to `Settings.environment`; pass `environment=` to choose another, or
`version="v7"` to pin one for a batch run or evaluation. The SDK caches each prompt for
`OPIK_PROMPT_CACHE_TTL_SECONDS` (300) and refreshes it in the background, so fetching on every
call, as above, costs nothing and picks up a moved label within five minutes.

An agent keeps the instructions it was built with, so a label move does not reach an agent
built once at startup; `run`, `run_sync`, and `run_stream` take `prompt=` for the trace link
and the result fields, not to change the instructions. Use `complete` where the prompt should
follow the label per call. A batch run fetches once at its start, builds its agent from that
version, and uses it throughout.

When nothing is cached and Opik does not answer within two seconds, `get_prompt` returns the
embedded default, and a run given it is marked degraded with `embedded_prompt` on the result
and the trace. A missing prompt, label, or version raises `LookupError`, and any other refusal
from Opik (a 401, say) raises the SDK's error: those are configuration errors, not an outage.
With tracing off (`OPIK_TRACK_DISABLE=true`) `get_prompt` returns the embedded default
unmarked, so local development needs no Opik.

Passing the prompt to `complete`, `run`, or `run_stream` links its version to the run's trace and puts `prompt_name` and
`prompt_version` on the result and in the trace metadata. A run that renders several managed
prompts, e.g. the instructions and a user message template, passes them as a list,
`prompt=[system, update]`: every version is linked, an embedded default among them marks the
run degraded, and the first names the result and the trace metadata. A call site that is not built on
PydanticAI gets the link by calling `get_prompt` inside its own `@opik.track`, and marks a
fallback itself with `degraded_metadata(prompt.degraded_reasons)`.

`ChatTemplate(messages=[...])` declares a chat prompt; text or chat is fixed per prompt name.
Templates are mustache (`{{key}}`, every placeholder must be given) unless declared
`type="jinja2"`. The fetched version and the embedded default render through the same SDK
code.

`prompt.template_hash` is a hash of the template text and type, which batch idempotency keys
are built on in place of the version. A run that fell back to the embedded default then gets
the same key as one that fetched the same text, so an Opik outage does not trigger
reprocessing.

The CI jobs use `ol_llm.create_prompt(name, template, project_name=...)`, which adds a version
only when the text or template type changed, and `ol_llm.promote_prompt(name, version="v7",
environment="production", project_name=...)`, which moves one label and leaves the version's
other labels in place. Opik seeds only `development`, `staging`, and `production`; create `ci`
and `qa` once per instance before promoting to them.

### Embeddings

An embedding call site is a use case too, naming an embedding model from the catalog. Like
`complete`, `ol_llm.embed` (or `ol_llm.embed_sync`) takes the use case and the texts:

```python
FEEDBACK_EMBEDDING = ol_llm.UseCase(
    app="ol-data-platform",
    name="feedback-embedding",
    profile="batch",
    model="bedrock:amazon.titan-embed-text-v2:0",
)

result = ol_llm.embed_sync(FEEDBACK_EMBEDDING, [summary], input_type="document")
result.embeddings  # one vector per text
result.model, result.dimensions  # store these with the vectors
```

The embedding model has the same timeout, retries, and `LLMError` classes as an agent's
model. It is built on the first call for a use case and reused after that. Each call is one
Opik trace with the embedding span and its token usage inside it. `input_type` is `query` or
`document` and has no default, because Google and Cohere models embed the two differently.
`dimensions=1024` asks a model that can shorten its vectors (the OpenAI `text-embedding-3`
models, Titan v2) for that length. A response that cannot be read, or that has fewer vectors
than texts, raises `LLMInvalidOutput`. The attribution's user id reaches OpenAI and Azure as
the request's `user` field; Bedrock and Google have no such field, so for them it is on the
trace only. An embedding use case takes no `fallback_models`, since
another model's vectors are not comparable with the primary's.

A call site built on LangChain or LangGraph keeps its own embedding client, since `embed`
would bring PydanticAI in beside it, and its OpenAI path needs `openai>=3.19`, which litellm
and llama-index-llms-openai do not allow yet. It wraps each provider request in
`ol_llm.traced_embedding`, which needs no extra:

```python
SEARCH_QUERY = ol_llm.UseCase(
    app="mit-learn",
    name="search-query",
    profile="request",
    model="openai:text-embedding-3-large",
)

with ol_llm.traced_embedding(SEARCH_QUERY, [query], attribution=attribution) as request:
    response = litellm.embedding(model="text-embedding-3-large", input=[query])
    cached = response._hidden_params.get("cache_hit") is True
    request.record(
        [item["embedding"] for item in response.data],
        input_tokens=0 if cached else response.usage.prompt_tokens,
    )
```

It runs the same catalog checks as `embed`, opens the same root trace, and records a span
shaped like `embed`'s with the model and the input tokens the caller reports. Report zero for
a response served wholly from a cache, since litellm fills in the usage of cached items.
litellm also marks a partial hit as a cache hit and sums cached and billed usage, so for
several texts count the misses' tokens or embed them without its cache. Read the response
before `to_dict()`, which drops `_hidden_params`. Wrap each request rather than a call that
hedges around several, so every billed request is counted; a request made on a worker
thread is its own root trace unless it is submitted with `contextvars.copy_context().run`.
The client must call the use case's model. Retries, timeouts (from `resolve_policy`), and
errors stay with the client: the block's exception passes through unchanged, and the trace
records its message only when `OL_LLM_TRACE_CONTENT` is on. A block that ends without
`record` raises `ValueError`.

Hedging, caching, batching texts to the provider's limits, and the vector store stay in the
application. Bedrock's Titan models take one text per request, so the library sends one
request per text, and a retry sends the whole call again: keep batches small enough that
repeating one is cheap.

Changing an embedding model is a catalog lifecycle event with a re-embedding plan, not a
settings change. Vectors are stored with the model id and dimension that produced them
(`EmbedResult.model` and `EmbedResult.dimensions`), as ol-data-platform's feedback embeddings
already are, so a new model or size adds vectors beside the old ones, and the application
switches its queries over once everything is re-embedded. The catalog entry records each
embedding model's default length, e.g. `ol_llm.load_catalog().get(model_id).dimensions` to
size a collection without an embedding call.

### Batch

`ol_llm.process_batch(rows, call, persist)` runs `call` on each row and hands every
`chunk_size` rows to `persist` as soon as they finish, so a crash loses one chunk at most.
A row whose call raises an `LLMError` goes in the failure ledger (`chunk.failed`) instead of
failing its chunk. Each `RowFailure` says whether the next run should try the row again:
rate limits, outages, and rejections of the account or model yes; invalid output, a content
filter, an exceeded usage limit, or a rejection of the row's input (a 400, 413, or 422) no,
because the same prompt and model would fail the same way. A `CircuitBreaker` stops the
batch after the chunk in which any other `LLMRejected` arrives (a 401, 403, 404, or a
transport error: our bug or configuration, which every later row would repeat), ten rows in
a row rejected for their input (a request every row gets rejected for), or three
rate-limited or unavailable rows in a row. The batch then raises `BatchFailed` with sample
errors, as it does when every attempted row failed, which would otherwise look like a run
with nothing to do. When rejections opened the circuit, or every attempted row was rejected
for its input (a batch too short to reach ten), `report.request_rejections` lists those
rows: they were marked not to retry, but the fix (e.g. dropping a parameter the model no
longer takes) does not change their key, so clear their skip marks.
Anything else a call raises is a bug and stops the batch unpersisted.

`ol_llm.BatchKeys(model=use_case.model, prompt=prompt)`, built from the prompt the run
fetched once at its start, makes the idempotency key from the row's input, the prompt name,
`prompt.template_hash`, and the model. A prompt or model change reprocesses the row, and an
Opik outage (which falls back to the same template) does not. The prompt version goes in
the row's `Provenance`, not the key. Store the key with the row and skip rows whose key
still matches.

A Dagster asset, where the pool limit set on the instance (`dagster instance concurrency set
llm/ol-data-platform/feedback-summary 2`) bounds the use case across assets:

```python
from dataclasses import asdict

import dagster as dg
import ol_llm

SUMMARY = ol_llm.UseCase(
    app="ol-data-platform",
    name="feedback-summary",
    profile="batch",
    model="bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
SUMMARY_PROMPT = ol_llm.TextTemplate(template=EXPORTED_TEXT, version="v3")


@dg.asset(pool=ol_llm.dagster_pool(SUMMARY))
def feedback_summaries(context: dg.AssetExecutionContext) -> None:
    prompt = ol_llm.get_prompt(
        SUMMARY.name, project_name=SUMMARY.app, default=SUMMARY_PROMPT
    )
    keys = ol_llm.BatchKeys(model=SUMMARY.model, prompt=prompt)
    stored = load_stored_keys()  # {conversation_pk: idempotency_key}
    rows = [
        row
        for row in load_conversations()
        if stored.get(row["pk"]) != keys.key(row["text"])
    ]
    attribution = ol_llm.Attribution(run_id=context.run_id)

    def summarize(row: dict) -> dict:
        result = ol_llm.complete_sync(
            SUMMARY,
            row["text"],
            instructions=prompt.format(),
            attribution=attribution,
            prompt=prompt,
        )
        provenance = keys.provenance(row["text"], result)
        return {"pk": row["pk"], "summary": result.output, **asdict(provenance)}

    def persist(chunk: ol_llm.Chunk) -> None:
        upsert_summaries([summary for _row, summary in chunk.succeeded])
        # Recorded so the next run skips them until the prompt or model changes.
        upsert_skipped(
            [
                failure.row["pk"]
                for failure in chunk.failed
                if not failure.retry_next_run
            ]
        )

    try:
        report = ol_llm.process_batch(
            rows, summarize, persist, max_workers=20, logger=context.log
        )
    except ol_llm.BatchFailed as exc:
        # The request, not these rows, was wrong; retry them once it is fixed.
        clear_skipped([failure.row["pk"] for failure in exc.report.request_rejections])
        raise dg.Failure(str(exc)) from exc
    context.add_output_metadata(
        {"summarized": report.succeeded, "failed": len(report.failures)}
    )
```

A Celery task that summarizes one slice of a group, with the rate limit counted in model
calls rather than task starts (Celery applies it per worker):

```python
from celery import shared_task
from django.db import transaction

import ol_llm

SUMMARY = ol_llm.UseCase(
    app="mit-learn", name="content-summary", profile="batch", model="openai:gpt-4o-mini"
)
SUMMARY_PROMPT = ol_llm.TextTemplate(template=EXPORTED_TEXT, version="v2")
FILES_PER_TASK = 20


@shared_task(
    bind=True,
    acks_late=True,
    rate_limit=ol_llm.celery_rate_limit(100, rows_per_task=FILES_PER_TASK, workers=2),
)
def summarize_content_files_task(self, content_file_ids: list[int]) -> int:
    prompt = ol_llm.get_prompt(
        SUMMARY.name, project_name=SUMMARY.app, default=SUMMARY_PROMPT
    )
    summarizer = ol_llm.build_agent(SUMMARY, instructions=prompt.format())
    keys = ol_llm.BatchKeys(model=SUMMARY.model, prompt=prompt)
    attribution = ol_llm.Attribution(run_id=self.request.id)
    files = [
        content_file
        for content_file in ContentFile.objects.filter(id__in=content_file_ids)
        if content_file.summary_key != keys.key(content_file.content)
    ]

    def summarize(content_file: ContentFile) -> ol_llm.LLMResult[str]:
        return ol_llm.run_sync(
            summarizer, content_file.content, attribution=attribution, prompt=prompt
        )

    def persist(chunk: ol_llm.Chunk) -> None:
        with transaction.atomic():
            for content_file, result in chunk.succeeded:
                content_file.summary = result.output
                content_file.summary_key = keys.key(content_file.content)
                content_file.save(update_fields=["summary", "summary_key"])

    return ol_llm.process_batch(files, summarize, persist, chunk_size=5).succeeded
```

The breaker lives in one task, so each task of a group has its own. A task time limit
(Celery `soft_time_limit`, Dagster `dagster/max_runtime`) has to sit above
`ol_llm.worst_case_seconds(use_case)`, the longest one model request can take under the
policy (3,300 s per model on the batch profile's defaults), or it kills a call that is still
retrying.

With `max_workers` above 1, rows run on threads, as in the Dagster example. `complete_sync`
and `run_sync` over an agent built once both work there, since each thread's event loop gets
its own provider clients. Threads do not inherit the caller's Opik context, so each row's run
is its own root trace.

### Tracing

Tracing reads the variables the Opik SDK reads, so the SDK and the span exporter always send
to the same instance:

| Variable | Meaning |
| --- | --- |
| `OPIK_URL_OVERRIDE` | the Opik API, e.g. `https://opik-qa.ol.mit.edu/api` |
| `OPIK_WORKSPACE` | `default` on our instances |
| `OPIK_KEYCLOAK_TOKEN_URL`, `OPIK_KEYCLOAK_CLIENT_ID`, `OPIK_KEYCLOAK_CLIENT_SECRET` | the Keycloak client the app authenticates to Opik with; `OPIK_KEYCLOAK_SCOPE` if it needs one. Leave `OPIK_API_KEY` unset |
| `OPIK_TRACK_DISABLE` | `true` turns tracing off, for local development without an Opik |
| `OL_LLM_TRACE_CONTENT` | `false` keeps prompts, completions, tool arguments, and error messages out of traces. On by default until the data-handling policy decides |

Without a call, tracing reads these the first time an agent or embedder is built or a run
starts. A host
that configures in code calls `ol_llm.configure_tracing(ol_llm.TracingConfig(...))` at
startup, before anything else touches the Opik SDK (the SDK reads the Keycloak hook when it
first builds its HTTP client) and before building agents (an agent keeps the instrumentation
it was built with). Once tracing has been used, a different configuration raises
`RuntimeError` and the one in effect stays. Short-lived processes should call
`ol_llm.tracing.flush()` before they exit. An app that registers its own Keycloak hook on the SDK (learn-ai's
`opik_keycloak_auth`) should drop it when it adopts the library.

### Testing

`ol_llm.testing.override_models` makes `build_model` and `build_agent` use PydanticAI's
`TestModel` or `FunctionModel` in place of a provider, still under the library's retry and
error handling. Set `OPIK_TRACK_DISABLE=true` for a test suite that should send nothing, or
use `ol_llm.testing.tracing_disabled` around one test. `opik.record_traces_locally()`
captures the root traces a test produces. Script responses and failures with
`FunctionModel`; do not commit recorded provider responses. `override_embedding_models` does
the same for `embed` and `embed_sync`, e.g. with PydanticAI's `TestEmbeddingModel`.

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
