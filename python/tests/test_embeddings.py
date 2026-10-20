import asyncio
import gc
import json
import threading
import weakref
from collections.abc import Callable, Generator, Sequence
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import opik
import pytest
from botocore.exceptions import ReadTimeoutError
from ol_llm import (
    Attribution,
    EmbedResult,
    LLMContentFiltered,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
    Settings,
    UseCase,
    configure,
    embed,
    embed_sync,
    models,
    tracing,
)
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.embeddings import PolicyEmbeddingModel, build_embedder, shared_embedder
from ol_llm.settings import GatewayRoute
from ol_llm.testing import override_embedding_models, tracing_disabled
from ol_llm.tracing import PROJECT_ATTRIBUTE, TracingConfig, configure_tracing
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic_ai.embeddings import EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)

MODEL = "openai:text-embedding-3-small"

SEARCH = UseCase(app="mit-learn", name="search-query", profile="request", model=MODEL)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")


class Scripted(TestEmbeddingModel):
    """Raises each scripted failure in turn, then embeds like TestEmbeddingModel."""

    def __init__(self, *failures: Exception) -> None:
        super().__init__(model_name="text-embedding-3-small", provider_name="openai")
        self.failures = list(failures)
        self.calls = 0

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return await super().embed(inputs, input_type=input_type, settings=settings)


class Short(TestEmbeddingModel):
    """Drops the last vector, as Google's model does for an empty one."""

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        result = await super().embed(inputs, input_type=input_type, settings=settings)
        result.embeddings = result.embeddings[:-1]
        return result


def http_error(status: int) -> ModelHTTPError:
    return ModelHTTPError(status, "text-embedding-3-small", body="provider said no")


def embed_async(*args: Any, **kwargs: Any) -> EmbedResult:  # noqa: ANN401
    return asyncio.run(embed(*args, **kwargs))


ENTRIES = [embed_sync, embed_async]


def embed_with(
    model: Any,  # noqa: ANN401
    texts: str | Sequence[str] = ("text",),
    *,
    entry: Callable[..., EmbedResult] = embed_sync,
    input_type: EmbedInputType = "document",
    **kwargs: Any,  # noqa: ANN401
) -> EmbedResult:
    with override_embedding_models(model):
        return entry(SEARCH, texts, input_type=input_type, **kwargs)


@pytest.mark.parametrize("entry", ENTRIES)
def test_embed_returns_the_vectors_model_and_dimension(
    entry: Callable[..., EmbedResult],
) -> None:
    result = embed_with(
        TestEmbeddingModel(dimensions=8), ["one text", "two"], entry=entry
    )

    assert len(result.embeddings) == 2
    assert result.model == MODEL
    assert result.dimensions == 8
    assert result.usage.input_tokens > 0


def test_a_single_text_is_one_input() -> None:
    result = embed_with(TestEmbeddingModel(), "a query", input_type="query")

    assert len(result.embeddings) == 1


def test_no_texts_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        embed_with(TestEmbeddingModel(), [])


def test_requested_dimensions_reach_the_model() -> None:
    model = TestEmbeddingModel(dimensions=1536)

    result = embed_with(model, dimensions=256)

    assert result.dimensions == 256
    assert model.last_settings == {"dimensions": 256}


@pytest.mark.usefixtures("provider_env")
def test_the_embedder_is_built_once_per_use_case_and_dimensions() -> None:
    first = shared_embedder(SEARCH)

    assert shared_embedder(SEARCH) is first
    assert shared_embedder(SEARCH, 256) is not first
    assert shared_embedder(SEARCH, 256) is shared_embedder(SEARCH, 256)


@pytest.mark.usefixtures("provider_env")
def test_the_embedder_is_rebuilt_when_the_settings_change() -> None:
    first = shared_embedder(SEARCH)
    configure(Settings(environment="qa"))

    assert shared_embedder(SEARCH) is not first


@pytest.mark.usefixtures("provider_env")
def test_an_override_bypasses_the_cache() -> None:
    cached = shared_embedder(SEARCH)

    with override_embedding_models(TestEmbeddingModel()):
        inside = shared_embedder(SEARCH)
        assert shared_embedder(SEARCH) is not inside
    assert inside is not cached
    assert shared_embedder(SEARCH) is cached


@pytest.mark.usefixtures("provider_env")
def test_each_event_loop_gets_its_own_provider_embedding_model() -> None:
    model = build_embedder(SEARCH).model
    assert isinstance(model, PolicyEmbeddingModel)

    async def loop_model() -> object:
        return model._loop_model()  # noqa: SLF001

    first, second = (asyncio.run(loop_model()) for _ in range(2))

    assert first is not second


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_root_trace_carries_the_attribution_fields(
    entry: Callable[..., EmbedResult],
) -> None:
    with opik.record_traces_locally() as storage:
        embed_with(
            TestEmbeddingModel(dimensions=8),
            ["a query"],
            entry=entry,
            input_type="query",
            attribution=ATTRIBUTION,
        )
        [trace] = storage.trace_trees

    assert trace.name == "search-query"
    assert trace.project_name == "mit-learn"
    assert trace.thread_id == "thread-1"
    assert sorted(trace.tags or []) == [
        "app:mit-learn",
        "env:ci",
        "use_case:search-query",
    ]
    assert trace.metadata is not None
    assert {k: trace.metadata[k] for k in ("use_case", "user_id", "run_id")} == {
        "use_case": "search-query",
        "user_id": "user-1",
        "run_id": "run-1",
    }
    assert trace.input == {"texts": ["a query"]}
    assert trace.output == {
        "model": MODEL,
        "dimensions": 8,
        "embeddings": 1,
        "input_tokens": 2,
    }


def test_the_embedding_span_lands_in_the_root_trace_with_its_usage(
    exported: InMemorySpanExporter,
) -> None:
    with opik.record_traces_locally() as storage:
        result = embed_with(TestEmbeddingModel("text-embedding-3-small"))
        [trace] = storage.trace_trees
    tracing.flush()

    [span] = exported.get_finished_spans()
    attributes = span.attributes or {}
    assert span.name == "embeddings text-embedding-3-small"
    assert attributes["opik.trace_id"] == trace.id
    assert attributes[PROJECT_ATTRIBUTE] == "mit-learn"
    assert attributes["gen_ai.usage.input_tokens"] == result.usage.input_tokens


def test_an_embedding_inside_a_tracked_function_is_a_span_of_its_trace(
    exported: InMemorySpanExporter,
) -> None:
    @opik.track(project_name="ol-data-platform")
    def handler() -> None:
        embed_with(TestEmbeddingModel())

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees
        [root] = storage.span_trees
    tracing.flush()

    assert trace.name == "handler"
    assert "use_case:search-query" in (trace.tags or [])
    assert [span.name for span in root.spans] == ["search-query"]
    [span] = exported.get_finished_spans()
    assert (span.attributes or {})[PROJECT_ATTRIBUTE] == "ol-data-platform"


def test_with_tracing_off_nothing_is_recorded(exported: InMemorySpanExporter) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))

    with opik.record_traces_locally() as storage:
        embed_with(TestEmbeddingModel())
        traces = storage.trace_trees

    assert traces == []
    with override_embedding_models(TestEmbeddingModel()):
        assert build_embedder(SEARCH).instrument is None
    assert exported.get_finished_spans() == ()


def test_with_tracing_disabled_at_runtime_nothing_is_recorded(
    exported: InMemorySpanExporter,
) -> None:
    with opik.record_traces_locally() as storage, tracing_disabled():
        embed_with(TestEmbeddingModel())
        traces = storage.trace_trees
    tracing.flush()

    assert traces == []
    assert exported.get_finished_spans() == ()


def test_a_rate_limit_is_retried(sleeps: list[float]) -> None:
    model = Scripted(http_error(429))

    result = embed_with(model)

    assert result.dimensions == 8
    assert model.calls == 2
    assert len(sleeps) == 1


def test_an_outage_spends_the_retries_and_raises_unavailable() -> None:
    model = Scripted(*(http_error(503) for _ in range(5)))

    with opik.record_traces_locally() as storage:
        with pytest.raises(LLMUnavailable) as caught:
            embed_with(model)
        [trace] = storage.trace_trees

    assert model.calls == 3
    assert caught.value.use_case == "search-query"
    assert caught.value.model == MODEL
    assert isinstance(caught.value.__cause__, ModelHTTPError)
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "LLMUnavailable"


def test_a_rejected_request_is_not_retried() -> None:
    model = Scripted(http_error(400))

    with pytest.raises(LLMRejected):
        embed_with(model, entry=embed_async)

    assert model.calls == 1


def test_a_raw_botocore_timeout_is_unavailable() -> None:
    # Bedrock's embedding model lets botocore's transport errors through raw.
    model = Scripted(ReadTimeoutError(endpoint_url="http://bedrock"))

    embed_with(model)

    assert model.calls == 2


@pytest.mark.parametrize(
    ("failures", "error", "calls"),
    [
        # Titan sends one request per text and raises its failures as a group.
        ((ExceptionGroup("titan", [http_error(429)]),) * 3, LLMRateLimited, 3),
        # Only retryable members: the first one decides.
        (
            (ExceptionGroup("titan", [http_error(503), http_error(429)]),) * 3,
            LLMUnavailable,
            3,
        ),
        (
            (ExceptionGroup("titan", [http_error(429), http_error(400)]),),
            LLMRejected,
            1,
        ),
    ],
)
def test_an_exception_group_is_classified_by_its_members(
    failures: tuple[Exception, ...], error: type[Exception], calls: int
) -> None:
    model = Scripted(*failures)

    with pytest.raises(error) as caught:
        embed_with(model, ["one", "two"])

    assert model.calls == calls
    assert isinstance(caught.value.__cause__, ExceptionGroup)


def test_a_group_with_an_unclassified_member_is_not_replaced() -> None:
    model = Scripted(ExceptionGroup("titan", [http_error(429), RuntimeError("bug")]))

    with pytest.raises(ExceptionGroup):
        embed_with(model)

    assert model.calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        UnexpectedModelBehavior("no embeddings field"),
        KeyError("embedding"),
        # Titan and Nova read each text's response inside a task group.
        ExceptionGroup("titan", [KeyError("embedding")]),
        ExceptionGroup("nova", [UnexpectedModelBehavior("no embedding field")]),
        ExceptionGroup("titan", [http_error(429), KeyError("embedding")]),
    ],
    ids=["unexpected", "key-error", "titan", "nova", "mixed"],
)
def test_a_malformed_response_raises_invalid_output(failure: Exception) -> None:
    model = Scripted(failure)

    with pytest.raises(LLMInvalidOutput) as caught:
        embed_with(model, ["one", "two"])

    assert model.calls == 1
    assert caught.value.model == MODEL
    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)
    assert caught.value.__cause__.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        ContentFilterError("blocked"),
        ExceptionGroup("titan", [http_error(429), ContentFilterError("blocked")]),
    ],
    ids=["direct", "grouped"],
)
def test_a_filtered_response_raises_content_filtered(failure: Exception) -> None:
    model = Scripted(failure)

    with pytest.raises(LLMContentFiltered) as caught:
        embed_with(model, ["one", "two"])

    assert model.calls == 1
    assert caught.value.__cause__ is not None
    assert caught.value.__cause__.__cause__ is failure


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_missing_vector_raises_invalid_output(
    entry: Callable[..., EmbedResult],
) -> None:
    with pytest.raises(LLMInvalidOutput, match="1 vectors for 2 texts"):
        embed_with(Short(), ["one", "two"], entry=entry)


@pytest.mark.parametrize(
    ("model_id", "attribution", "sent"),
    [
        (MODEL, ATTRIBUTION, {"extra_body": {"user": "user-1"}}),
        (
            "azure:text-embedding-3-large",
            ATTRIBUTION,
            {"extra_body": {"user": "user-1"}},
        ),
        (MODEL, Attribution(run_id="run-1"), None),
        ("bedrock:amazon.titan-embed-text-v2:0", ATTRIBUTION, None),
    ],
)
def test_the_user_id_reaches_providers_with_a_user_field(
    model_id: str, attribution: Attribution, sent: dict[str, Any] | None
) -> None:
    model = TestEmbeddingModel()
    use_case = UseCase(
        app="mit-learn", name="search", profile="request", model=model_id
    )

    with override_embedding_models(model):
        embed_sync(use_case, ["q"], input_type="query", attribution=attribution)

    assert model.last_settings == (sent or {})


def test_embed_refuses_a_chat_model() -> None:
    chat = UseCase(app="a", name="b", profile="request", model="openai:gpt-4o-mini")

    with pytest.raises(ValueError, match="kind 'chat', not 'embedding'"):
        embed_sync(chat, ["text"], input_type="query")


def test_an_unlisted_model_embeds_and_tags_its_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unlisted = UseCase(
        app="a", name="b", profile="batch", model="openai:text-embedding-ada-002"
    )

    with (
        override_embedding_models(TestEmbeddingModel(dimensions=8)),
        opik.record_traces_locally() as storage,
    ):
        result = embed_sync(unlisted, ["text"], input_type="document")
        [trace] = storage.trace_trees

    assert result.dimensions == 8
    assert "model_status:uncataloged" in (trace.tags or [])
    assert "the model catalog does not list" in caplog.text


def test_build_embedder_refuses_fallback_models() -> None:
    with_fallback = UseCase(
        app="a",
        name="b",
        profile="batch",
        model=MODEL,
        fallback_models=("azure:text-embedding-3-small",),
    )

    with pytest.raises(ValueError, match="not comparable"):
        build_embedder(with_fallback)


def test_build_embedder_refuses_a_provider_it_does_not_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohere = CatalogEntry(
        id="cohere:embed-v4.0",
        kind="embedding",
        dimensions=1536,
        input_usd_per_mtok=Decimal("0.12"),
        output_usd_per_mtok=Decimal(0),
    )
    catalog = Catalog({**load_catalog().entries, cohere.id: cohere})
    monkeypatch.setattr(models, "load_catalog", lambda: catalog)

    with pytest.raises(ValueError, match="does not build 'cohere' embedding"):
        build_embedder(UseCase(app="a", name="b", profile="batch", model=cohere.id))


def test_the_embedder_model_carries_the_policy() -> None:
    with override_embedding_models(TestEmbeddingModel()):
        model = build_embedder(SEARCH).model

    assert isinstance(model, PolicyEmbeddingModel)
    assert model.catalog_entry is load_catalog().entries[MODEL]
    assert model.policy.max_retries == 2


# A real OpenAI client against a local server that keeps connections alive, so a
# second call reuses the first call's pooled connection.

EMBEDDINGS_REPLY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


class KeepAliveProvider(BaseHTTPRequestHandler):
    """Answers OpenAI embeddings requests over persistent connections."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps(EMBEDDINGS_REPLY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def real_client(provider_env: None) -> Generator[None]:  # noqa: ARG001
    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveProvider)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    route = GatewayRoute(base_url=url, api_key="app-key")
    configure(Settings(environment="ci", gateway_routes={"openai": route}))
    yield
    server.shutdown()
    server.server_close()


@pytest.mark.usefixtures("real_client")
def test_embed_sync_works_from_several_threads() -> None:
    dimensions: list[int] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            result = embed_sync(SEARCH, ["hi"], input_type="query")
            dimensions.append(result.dimensions)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(2):
        thread = threading.Thread(target=call)
        thread.start()
        thread.join()

    assert errors == []
    assert dimensions == [3, 3]


@pytest.mark.usefixtures("real_client")
def test_a_finished_threads_loop_is_released() -> None:
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    def call() -> None:
        embed_sync(SEARCH, ["hi"], input_type="query")
        loops.append(weakref.ref(asyncio.get_event_loop()))

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    gc.collect()

    assert loops[0]() is None


@pytest.mark.usefixtures("real_client")
def test_embed_works_under_successive_event_loops() -> None:
    results = [asyncio.run(embed(SEARCH, ["hi"], input_type="query")) for _ in range(2)]

    assert [result.dimensions for result in results] == [3, 3]
